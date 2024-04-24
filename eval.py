import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import set_seed
from dataset import MolGenDataModule
from models import GPT2MolGen, GPT2MolGen_flash_atten, Llama_small_flash_atten
from utils import creat_unique_experiment_name, is_world_process_zero
import wandb
import torch
import os
from multiprocessing import Pool
from typing import List, Tuple
from hydra import compose, initialize
from trainer import MyHFTrainer, MyTrainingArguments
from callbacks import WandbCallback
from fcd_torch import FCD as FCDMetric
import argparse

import pandas as pd
from moses.dataset import get_dataset, get_statistics
from moses.metrics.metrics import (
    FragMetric,
    ScafMetric,
    SNNMetric,
    WassersteinMetric,
    compute_intermediate_statistics,
    fraction_passes_filters,
    fraction_unique,
    fraction_valid,
    internal_diversity,
    novelty,
    remove_invalid,
)
from moses.metrics.utils import QED, SA, get_mol, logP, weight
from moses.utils import disable_rdkit_log, enable_rdkit_log, mapper
from tabulate import tabulate
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, LlamaForCausalLM, LlamaConfig
from tqdm import tqdm
import copy

MOLECULAR_PERFORMANCE_RESULT_PATH = './molecular_performance_MOSES.csv'

def args_parser():
    parser = argparse.ArgumentParser(
        description='Molecules Generation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config_name', default="train_moses_llama", type=str,
                        help='name of the trained config')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42],
                        help='list of seed integers for random number generation')
    parser.add_argument('--num_samples', default=30000, type=int, help='number of samples to generate')
    parser.add_argument('--batch_size', default=1024, type=int, help='batch size')
    parser.add_argument('--top_k', default=50, type=int, help='top_k')
    parser.add_argument('--top_p', default=0.95, type=float, help='top_p')
    parser.add_argument('--temperature', default=1.0, type=float, help='temperature')
    parser.add_argument('--prompt', default="", type=str, help='input prompt to generate')
    parser.add_argument('--preprocess_num_jobs', default=24, type=int, help='preprocess_num_jobs')
    args = parser.parse_args()
    return args

def filter_tokens_after_eos(sequences, eos_id):
    output = copy.deepcopy(sequences)
    for i in range(sequences.size(0)):
        row = sequences[i]
        eos_position = (row == eos_id).nonzero()
        if eos_position.numel() > 0:
            eos_position = eos_position[0, 0].item()  # Get the index of the first occurrence
            output[i, eos_position+1:] = eos_id
    return output

def generate_smiles_FA(
        model,
        tokenizer,
        n_samples: int = 30000,
        num_return_sequences: int = 1000,
        prompt: str = "CC",
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        max_length: int = 64,
        device: torch.device = torch.device('cuda'),
):
    """Generate SMILES from model.
    :param max_length:
    :param num_return_sequences:
    :param device:
    :param model: GPT2 model
    :param tokenizer: GPT2 tokenizer
    :param n_samples: Number of samples to generate
    :param prompt: Prompt to start generation
    :param temperature: Temperature for sampling
    :param top_k: Top K for sampling
    :param top_p: Top p for sampling
    :return: List of SMILES
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_ids = input_ids.repeat_interleave(num_return_sequences, dim=0)
    # skip eos token
    input_ids = input_ids[:, :-1]
    generated_sequences = []
    for i in tqdm(range(n_samples // num_return_sequences)):
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
        generated_sequences += [s.replace(" ", "") for s in output]

    return generated_sequences


def generate_smiles_HF(
        model,
        tokenizer,
        n_samples: int = 30000,
        num_return_sequences: int = 32,
        no_repeat_ngram_size: int = 2,
        prompt: str = 'CC',
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        max_length: int = 64,
        device: torch.device = torch.device('cuda'),
):
    """Generate SMILES from model.

        :param max_length:
        :param num_return_sequences:
        :param device:
        :param no_repeat_ngram_size:
        :param model: GPT2 model
        :param tokenizer: GPT2 tokenizer
        :param n_samples: Number of samples to generate
        :param prompt: Prompt to start generation
        :param temperature: Temperature for sampling
        :param top_k: Top K for sampling
        :param top_p: Top p for sampling
        :return: List of SMILES
        """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    # skip eos token
    input_ids = input_ids[:, :-1]
    generated_sequences = []
    for i in tqdm(range(n_samples // num_return_sequences)):
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=no_repeat_ngram_size,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )
        output = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        generated_sequences += [s.replace(" ", "") for s in output]

    return generated_sequences


def get_all_metrics(
        gen,
        k=None,
        n_jobs=1,
        device="cpu",
        batch_size=512,
        pool=None,
        test=None,
        test_scaffolds=None,
        ptest=None,
        ptest_scaffolds=None,
        train=None,
        report=['all'],
):
    if test is None:
        if ptest is not None:
            raise ValueError("You cannot specify custom test " "statistics for default test set")
        test = get_dataset("test")
        ptest = get_statistics("test")

    if test_scaffolds is None:
        if ptest_scaffolds is not None:
            raise ValueError(
                "You cannot specify custom scaffold test "
                "statistics for default scaffold test set"
            )
        test_scaffolds = get_dataset("test_scaffolds")
        ptest_scaffolds = get_statistics("test_scaffolds")

    if train is None:
        train = get_dataset("train")

    if k is None:
        k = [1000, 10000]
    disable_rdkit_log()
    metrics = {}
    close_pool = False
    if pool is None:
        if n_jobs != 1:
            pool = Pool(n_jobs)
            close_pool = True
        else:
            pool = 1
    if "valid" in report or "all" in report:
        metrics["valid"] = fraction_valid(gen, n_jobs=pool)  # type: ignore

    if "unique@" in report or "all" in report:
        gen = remove_invalid(gen, canonize=True)
        if not isinstance(k, (list, tuple)):
            k = [k]
        for _k in k:
            metrics[f"unique@{_k}"] = fraction_unique(gen, _k, pool)  # type: ignore

    if "SNN" in report or "all" in report:
        if ptest is None:
            ptest = compute_intermediate_statistics(
                test, n_jobs=n_jobs, device=device, batch_size=batch_size, pool=pool
            )
        if test_scaffolds is not None and ptest_scaffolds is None:
            ptest_scaffolds = compute_intermediate_statistics(
                test_scaffolds,
                n_jobs=n_jobs,
                device=device,
                batch_size=batch_size,
                pool=pool,
            )
        mols = mapper(pool)(get_mol, gen)
        kwargs = {"n_jobs": pool, "device": device, "batch_size": batch_size}
        kwargs_fcd = {'n_jobs': n_jobs, 'device': device, 'batch_size': batch_size}
        metrics['FCD/Test'] = FCDMetric(**kwargs_fcd)(gen=gen, pref=ptest['FCD'])
        metrics["SNN/Test"] = SNNMetric(**kwargs)(gen=mols, pref=ptest["SNN"])
        metrics["Frag/Test"] = FragMetric(**kwargs)(gen=mols, pref=ptest["Frag"])
        metrics["Scaf/Test"] = ScafMetric(**kwargs)(gen=mols, pref=ptest["Scaf"])
        if ptest_scaffolds is not None:
            metrics['FCD/TestSF'] = FCDMetric(**kwargs_fcd)(gen=gen, pref=ptest_scaffolds['FCD'])
            metrics["SNN/TestSF"] = SNNMetric(**kwargs)(gen=mols, pref=ptest_scaffolds["SNN"])
            metrics["Frag/TestSF"] = FragMetric(**kwargs)(gen=mols, pref=ptest_scaffolds["Frag"])
            metrics["Scaf/TestSF"] = ScafMetric(**kwargs)(gen=mols, pref=ptest_scaffolds["Scaf"])

    if "IntDiv" in report or "all" in report:
        metrics["IntDiv"] = internal_diversity(mols, pool, device=device)  # type: ignore
        metrics["IntDiv2"] = internal_diversity(mols, pool, device=device, p=2)  # type: ignore
        metrics["Filters"] = fraction_passes_filters(mols, pool)  # type: ignore

    if "logP" in report or "all" in report:
        # Properties
        for name, func in [("logP", logP), ("SA", SA), ("QED", QED), ("weight", weight)]:
            metrics[name] = WassersteinMetric(func, **kwargs)(gen=mols, pref=ptest[name])

    if "Novelty" in report or "all" in report:
        if train is not None:
            metrics["Novelty"] = novelty(mols, train, pool)  # type: ignore
    enable_rdkit_log()
    if close_pool:
        pool.close()  # type: ignore
        pool.join()  # type: ignore
    return metrics

def get_spec_metrics(
        gen,
        n_jobs=1,
        device="cpu",
        batch_size=512,
        pool=None,
        test=None,
        test_scaffolds=None,
        ptest=None,
        ptest_scaffolds=None,
        train=None,
):
    if test is None:
        if ptest is not None:
            raise ValueError("You cannot specify custom test " "statistics for default test set")
        test = get_dataset("test")
        ptest = get_statistics("test")

    if test_scaffolds is None:
        if ptest_scaffolds is not None:
            raise ValueError(
                "You cannot specify custom scaffold test "
                "statistics for default scaffold test set"
            )

    if train is None:
        train = get_dataset("train")

    disable_rdkit_log()
    metrics = {}
    close_pool = False
    if pool is None:
        if n_jobs != 1:
            pool = Pool(n_jobs)
            close_pool = True
        else:
            pool = 1
    metrics["valid"] = fraction_valid(gen, n_jobs=pool)  # type: ignore

    gen = remove_invalid(gen, canonize=True)
    metrics[f"unique@{1000}"] = fraction_unique(gen, 1000, pool)  # type: ignore

    if ptest is None:
        ptest = compute_intermediate_statistics(
            test, n_jobs=n_jobs, device=device, batch_size=batch_size, pool=pool
        )
    mols = mapper(pool)(get_mol, gen)
    kwargs = {"n_jobs": pool, "device": device, "batch_size": batch_size}
    kwargs_fcd = {'n_jobs': n_jobs, 'device': device, 'batch_size': batch_size}
    metrics['FCD/Test'] = FCDMetric(**kwargs_fcd)(gen=gen, pref=ptest['FCD'])
    metrics["SNN/Test"] = SNNMetric(**kwargs)(gen=mols, pref=ptest["SNN"])
    metrics["Frag/Test"] = FragMetric(**kwargs)(gen=mols, pref=ptest["Frag"])
    metrics["Scaf/Test"] = ScafMetric(**kwargs)(gen=mols, pref=ptest["Scaf"])
    metrics["IntDiv"] = internal_diversity(mols, pool, device=device)  # type: ignore

    if train is not None:
        metrics["Novelty"] = novelty(mols, train, pool)  # type: ignore
    if close_pool:
        pool.close()  # type: ignore
        pool.join()  # type: ignore
    return metrics


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
        print(f" ============= start generate for seed={seed} =============")
        set_seed(seed)
        if isinstance(model, Llama_small_flash_atten) or isinstance(model, GPT2MolGen_flash_atten):
            generated_smiles = generate_smiles_FA(
                model=model,
                tokenizer=datamodule.tokenizer,
                n_samples=args.n_samples,
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
                n_samples=args.n_samples,
                num_return_sequences=args.batch_size,
                prompt=args.prompt,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                max_length=datamodule.max_seq_length,
                device=torch.device('cuda')
            )

        # Save the SMILES to a CSV file
        df = pd.DataFrame(generated_smiles, columns=["SMILES"])
        df.to_csv(os.path.join(output_dir, f'generated_smiles_{seed}.csv'), index=False)

        # compute metrics
        metrics = get_spec_metrics(generated_smiles, n_jobs=cfg.eval.preprocess_num_jobs)
        metrics_table = [[k, v] for k, v in metrics.items()]
        print(tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="pretty"))

        save_path = os.path.join(output_dir, MOLECULAR_PERFORMANCE_RESULT_PATH)
        if os.path.exists(save_path):
            result = pd.read_csv(save_path, index_col=0)
        else:
            result = pd.DataFrame()
        df_new_row = pd.DataFrame(metrics, index=[])
        result = pd.concat([result, df_new_row])
        result.to_csv(save_path)


if __name__ == "__main__":
    args = args_parser()
    entrypoint(args)
