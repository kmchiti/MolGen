from transformers import set_seed
from dataset import MolGenDataModule
from models import GPT2MolGen, GPT2MolGen_flash_atten, Llama_small_flash_atten
from utils import creat_unique_experiment_name, is_world_process_zero
import wandb
import numpy as np
import torch
import os
from hydra import compose, initialize
from trainer import MyHFTrainer, MyTrainingArguments
import argparse
import pandas as pd
from moses.metrics.utils import QED, SA, get_mol, logP, weight
from moses.utils import disable_rdkit_log,  mapper
from moses.metrics.metrics import canonic_smiles
from pandarallel import pandarallel
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, LlamaForCausalLM, LlamaConfig
from eval import generate_smiles_FA, generate_smiles_HF

MOLECULAR_PERFORMANCE_RESULT_PATH = './molecular_performance_MOSES.csv'
MOLECULAR_PROPERTY_RESULT_PATH = './molecular_property.csv'
PYTDC_RESULT_PATH = 'PyTDC_results'


def args_parser():
    parser = argparse.ArgumentParser(
        description='Molecules Generation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config_name', default="train_moses_llama", type=str,
                        help='name of the trained config')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42],
                        help='list of seed integers for random number generation')
    parser.add_argument('--num_samples', default=30000, type=int, help='number of samples to generate')
    parser.add_argument('--total_samples', default=500000, type=int, help='number of total samples to generate and filter')
    parser.add_argument('--batch_size', default=1024, type=int, help='batch size')
    parser.add_argument('--top_k', default=50, type=int, help='top_k')
    parser.add_argument('--top_p', default=0.95, type=float, help='top_p')
    parser.add_argument('--temperature', default=1.0, type=float, help='temperature')
    parser.add_argument('--prompt', default="", type=str, help='input prompt to generate')
    parser.add_argument('--preprocess_num_jobs', default=6, type=int, help='preprocess_num_jobs')
    parser.add_argument('--task', default='qed', type=str, help='task to filter for')
    args = parser.parse_args()
    return args


def get_qed(input_smiles):
    mol = get_mol(input_smiles)
    return QED(mol)

def get_logp(input_smiles):
    mol = get_mol(input_smiles)
    return logP(mol)

def get_sa(input_smiles):
    mol = get_mol(input_smiles)
    return SA(mol)


def entrypoint(args):
    # Initialize setup
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name=args.config_name)

    # Initialize setup
    exp_name = creat_unique_experiment_name(cfg)
    output_dir = os.path.join(cfg.save_path, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize DataModule
    datamodule = MolGenDataModule(**cfg.dataset)

    # Initialize model
    if cfg.model.model_name_or_path == 'gpt2_flash_atten':
        model = GPT2MolGen_flash_atten(**cfg.model, max_seq_length=datamodule.max_seq_length,
                                       vocab_size=datamodule.tokenizer.vocab_size)
    elif cfg.model.model_name_or_path in ['llama_small', 'llama_small_HF']:
        model_cfg = LlamaConfig(**cfg.model, n_ctx=datamodule.max_seq_length,
                                vocab_size=datamodule.tokenizer.vocab_size)
        model = LlamaForCausalLM(model_cfg)
    elif cfg.model.model_name_or_path == 'llama_small_FA':
        model = Llama_small_flash_atten(**cfg.model, max_seq_length=datamodule.max_seq_length,
                                        vocab_size=datamodule.tokenizer.vocab_size)
    elif cfg.model.model_name_or_path == 'gpt2':
        model = GPT2MolGen(**cfg.model, max_seq_length=datamodule.max_seq_length,
                           vocab_size=datamodule.tokenizer.vocab_size)
    else:
        raise NotImplementedError

    # Initialize trainer
    wandb_callback = None
    train_args = MyTrainingArguments(data_seed=cfg.seed, seed=cfg.seed, output_dir=output_dir, **cfg.trainer)

    trainer = MyHFTrainer(model=model,
                          args=train_args,
                          callbacks=wandb_callback,
                          tokenizer=datamodule.tokenizer,
                          data_collator=datamodule.data_collator,
                          train_dataset=datamodule.train_dataset,
                          eval_dataset=datamodule.eval_dataset,
                          evaluation_task=None,
                          )

    print(f"load checkpoint from: {output_dir}")
    trainer._load_from_checkpoint(output_dir)
    model = trainer.accelerator.prepare_model(model, evaluation_mode=True)

    # Generate SMILES and calculate metrics
    model.eval()
    for seed in args.seeds:
        set_seed(seed)
        print(f" ============= start generate for seed={seed} =============")
        final_df = pd.DataFrame()
        while len(final_df) <= args.total_samples:
            if isinstance(model, Llama_small_flash_atten) or isinstance(model, GPT2MolGen_flash_atten):
                generated_smiles = generate_smiles_FA(
                    model=model,
                    tokenizer=datamodule.tokenizer,
                    n_samples=args.num_samples,
                    num_return_sequences=args.batch_size,
                    prompt=args.prompt,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    max_length=datamodule.max_seq_length,
                    device=torch.device('cuda')
                )
            else:
                generated_smiles = generate_smiles_HF(
                    model=model,
                    tokenizer=datamodule.tokenizer,
                    n_samples=args.num_samples,
                    num_return_sequences=args.batch_size,
                    prompt=args.prompt,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    max_length=datamodule.max_seq_length,
                    device=torch.device('cuda')
                )

            disable_rdkit_log()
            # remove invalid molecules
            gen_smiles = [x for x in mapper(args.preprocess_num_jobs)(canonic_smiles, generated_smiles) if x is not None]
            # select unique and valid molecules
            gen_smiles = list(set(gen_smiles) - {None})

            df = pd.DataFrame(gen_smiles, columns=["SMILES"])
            if args.task == 'qed':
                pandarallel.initialize(shm_size_mb=60720, nb_workers=args.args.preprocess_num_jobs, progress_bar=True)
                df['qed'] = df['SMILES'].parallel_apply(get_qed)
                filtered_df = df[df['qed'] > 0.9]
            elif args.task == 'logp':
                pandarallel.initialize(shm_size_mb=60720, nb_workers=args.args.preprocess_num_jobs, progress_bar=True)
                df['qed'] = df['SMILES'].parallel_apply(get_qed)
                filtered_df = df[df['qed'] > 0.9]
            else:
                raise NotImplementedError

            final_df = pd.concat([final_df, filtered_df])

        # Save the SMILES to a CSV file
        final_df.to_csv(os.path.join(output_dir, f'generated_smiles_{args.task}_{seed}.csv'), index=False)


if __name__ == "__main__":
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    args = args_parser()
    entrypoint(args)
