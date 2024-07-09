from dataset import MolGenDataModule
from models import GPT2MolGen, GPT2MolGen_flash_atten, Llama_small_flash_atten
from utils import creat_unique_experiment_name, is_world_process_zero
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, LlamaForCausalLM, LlamaConfig, set_seed
from accelerate import Accelerator
from hydra import compose, initialize
import numpy as np
import wandb
import torch
import os
import argparse
import pandas as pd
import torch.nn.functional as F

import copy
import time
import sys
from docking_score import DockingConfig, DockingVina
import tdc


def filter_tokens_after_eos(sequences, eos_id):
    output = copy.deepcopy(sequences)
    for i in range(sequences.size(0)):
        row = sequences[i]
        eos_position = (row == eos_id).nonzero()
        if eos_position.numel() > 0:
            eos_position = eos_position[0, 0].item()  # Get the index of the first occurrence
            output[i, eos_position + 1:] = eos_id
    return output


def generated_samples(
        model, tokenizer,
        batch_size: int = 64,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        max_length: int = 64,
        device: torch.device = torch.device('cuda'), ):
    prompt = ""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_ids = input_ids.repeat_interleave(batch_size, dim=0)
    # skip eos token
    input_ids = input_ids[:, :-1]

    output = model.generate(
        input_ids,
        max_length=max_length,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
    )
    sequences = filter_tokens_after_eos(output.sequences, eos_id=tokenizer.eos_token_id)
    output = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    generated_smiles = [s.replace(" ", "") for s in output]
    return sequences, generated_smiles


def get_reward(smiles, target='fa7', num_proc=24, seed=42, device=torch.device('cuda')):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    docking_cfg = DockingConfig(target_name=target, num_sub_proc=num_proc,
                                num_cpu_dock=1, seed=seed)
    target = DockingVina(docking_cfg)
    st = time.time()
    try:
        smiles_scores = target.predict(smiles)
    except Exception as e:
        print(f'FAILED: {str(e)}')
        sys.exit(1)
    print(f'finish docking in {time.time() - st} seconds')
    target.__del__()

    smiles_scores = torch.tensor(smiles_scores)
    return smiles_scores.to(device)


def compute_chemical_prop(generated_smiles, n_jobs=1):
    SA_scores = tdc.Oracle(name="SA")(generated_smiles)
    QED_scores = tdc.Oracle(name="qed")(generated_smiles)

    return torch.tensor(SA_scores), torch.tensor(QED_scores)


def args_parser():
    parser = argparse.ArgumentParser(
        description='Binding Affinity Score',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config_name', default="train_ZINC_270M_atomwise", type=str,
                        help='name of the trained config')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--num_oracle', default=40000, type=int, help='batch size')
    parser.add_argument('--target', default='fa7', type=str, help='task to filter for')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--wandb', action='store_true', help='use wandb for logging')
    parser.add_argument('--num_proc', default=8, type=int, help='number of processor')
    args = parser.parse_args()
    return args


def train(num_oracle, datamodule, model, optimizer, batch_size, num_proc, target='fa7', use_wandb=False):
    for i in range(num_oracle // batch_size):
        sequences, generated_smiles = generated_samples(model, datamodule.tokenizer,
                                                        batch_size=batch_size,
                                                        max_length=datamodule.max_seq_length)
        smiles_scores = get_reward(generated_smiles, target=target, num_proc=num_proc)

        SA_scores, QED_scores = compute_chemical_prop(generated_smiles, n_jobs=num_proc)
        num_invalid = (smiles_scores > 0).sum().item()
        top_10 = smiles_scores[smiles_scores < 0].sort()[0][:10].mean().item()
        smiles_scores[smiles_scores == 99.9] = 0
        reward = smiles_scores / (-20.)

        metrics = {'num_invalid': num_invalid, 'top_10': top_10, 'mean_reward': reward.mean().item(),
                   'max_reward': reward.max().item(), 'num_oracles': i * batch_size,
                   'SA_score': SA_scores.mean().item(), 'QED_score': QED_scores.mean().item()}

        logits = model(sequences).logits
        mask = (sequences == datamodule.tokenizer.eos_token_id).float()
        # To create a cumulative mask that only marks positions after the first EOS
        shifted_mask = torch.cat([torch.zeros_like(mask[:, :1]), mask[:, :-1]], dim=1)
        nonterms = ~(shifted_mask.cumsum(dim=1) > 0)
        nonterms[:, 0] = 0  # bos

        log_probs = F.log_softmax(logits, dim=-1) * nonterms.unsqueeze(-1)
        token_indexes = sequences.unsqueeze(-1)
        selected_log_probs = torch.gather(log_probs, dim=2, index=token_indexes)
        selected_log_probs = selected_log_probs.squeeze(-1)
        molecule_log_prob = selected_log_probs.sum(dim=1)

        policy_gradient = (reward * molecule_log_prob).mean()

        loss = -policy_gradient  # Negative for maximizing
        metrics['loss'] = loss.item()
        print(metrics)
        if use_wandb:
            wandb.log(metrics)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = args_parser()
    set_seed(args.seed)

    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name=args.config_name)
    exp_name = creat_unique_experiment_name(cfg)
    output_dir = os.path.join(cfg.save_path, exp_name)
    datamodule = MolGenDataModule(**cfg.dataset)
    model = Llama_small_flash_atten(**cfg.model, max_seq_length=datamodule.max_seq_length,
                                    vocab_size=datamodule.tokenizer.vocab_size)

    print(f"load checkpoint from {os.path.join(output_dir, 'pytorch_model.bin')}")
    checkpoint = torch.load(os.path.join(output_dir, 'pytorch_model.bin'), map_location='cpu')
    model.load_state_dict(checkpoint)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    accelerator = Accelerator(mixed_precision='bf16', device_placement='cuda')
    model, optimizer = accelerator.prepare(model, optimizer)

    if args.wandb:
        wandb.init(entity='drug-discovery', project='small-molecule-generation',
                   name=f'reinforce_{exp_name}_{args.target}',
                   config=args.__dict__)

    train(args.num_oracle, datamodule, model, optimizer, args.batch_size, args.num_proc,
          target=args.target, use_wandb=args.wandb)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    output_dir = os.path.join(output_dir, f"reinforce_{args.target}")
    os.makedirs(output_dir, exist_ok=True)
    check_path = os.path.join(output_dir, f"pytorch_model.tar")
    print(f'=============== save checkpoint in {check_path} ===============')
    torch.save(checkpoint, check_path)
