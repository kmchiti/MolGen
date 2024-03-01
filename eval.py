import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import set_seed, Trainer, TrainingArguments
from dataset import MolGenDataModule
from models import GPT2MolGen, GPT2MolGen_flash_atten
from utils import creat_unique_experiment_name, is_world_process_zero, save_HF_model
import wandb
import torch
import os
import argparse
from multiprocessing import Pool
from typing import List, Tuple

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
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from tqdm import tqdm

def _checkpoint_is_available(trainer):
    items = os.listdir(trainer.args.output_dir)
    checkpoint_found = any(
        item.startswith("checkpoint") and os.path.isdir(os.path.join(trainer.args.output_dir, item)) for item
        in items)
    return checkpoint_found

"""
Most frequent tokens:
[295] --> O=C(O)
[224] --> COC(=O)
[130] --> CC(C)
But in this code [69] --> CC was prefered.
"""
def generate_smiles_FA(
    model: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizerFast,
    n_samples: int,
    prompt: str,
    temperature: float,
    top_k: int,
    top_p: float,
    device: torch.device,
) -> List[str]:
    """Generate SMILES from model.
    This generation only works for batch = 1

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
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"][0][:-1].reshape(1, -1)
    outputs = []
    for i in tqdm(range(n_samples)):
        output = model.generate(input_ids=input_ids, max_length=64, temperature=temperature,
                                top_k=top_k, top_p=top_p,
                                eos_token_id=tokenizer.eos_token_id)
        output = tokenizer.batch_decode(output, skip_special_tokens=True)[0].replace(" ", "")
        outputs.append(output)
    return outputs


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
    device: torch.device = torch.device('cuda'),
):
    """Generate SMILES from model.

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
    for i in tqdm(range(n_samples//num_return_sequences)):
        output = model.generate(
                                input_ids,
                                max_length=64,
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
    report: List[str] = ['all'],
):
    """Computes all available metrics between test (scaffold test) and generated sets of SMILES.

    :param report:
    :param gen: list of generated SMILES
    :param k: int or list with values for unique@k. Will calculate number of unique molecules in the first k molecules. Default [1000, 10000]
    :param n_jobs: number of workers for parallel processing
    :param device: 'cpu' or 'cuda:n', where n is GPU device number
    :param batch_size: batch size for FCD metric
    :param pool: optional multiprocessing pool to use for parallelization
    :param test: test SMILES. If None, will load a default test set
    :param test_scaffolds: scaffold test SMILES. If None, will load a default scaffold test set
    :param ptest: precalculated statistics of the test set. If None, will load default test statistics. If you specified a custom test set, default test statistics will be ignored
    :param ptest_scaffolds: precalculated statistics of the scaffold test set If None, will load default scaffold test statistics. If you specified a custom test set, default test statistics will be ignored
    :param train: train SMILES. If None, will load a default train set

    Available metrics:
        * %valid
        * %unique@k
        * Fragment similarity (Frag)
        * Scaffold similarity (Scaf)
        * Similarity to nearest neighbour (SNN)
        * Internal diversity (IntDiv)
        * Internal diversity 2: using square root of mean squared Tanimoto similarity (IntDiv2)
        * %passes filters (Filters)
        * Distribution difference for logP, SA, QED, weight
        * Novelty (molecules not present in train)
    """
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

    train = train or get_dataset("train")

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
        metrics["SNN/Test"] = SNNMetric(**kwargs)(gen=mols, pref=ptest["SNN"])
        metrics["Frag/Test"] = FragMetric(**kwargs)(gen=mols, pref=ptest["Frag"])
        metrics["Scaf/Test"] = ScafMetric(**kwargs)(gen=mols, pref=ptest["Scaf"])
        if ptest_scaffolds is not None:
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


@hydra.main(version_base=None, config_path="configs", config_name="config_eval")
def entrypoint(cfg: DictConfig):

    # Initialize setup
    set_seed(cfg.seed)
    exp_name = creat_unique_experiment_name(cfg)
    output_dir = os.path.join(cfg.save_path, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize DataModule
    datamodule = MolGenDataModule(**cfg.dataset)

    # Load model
    if cfg.model.model_name_or_path == 'gpt2_flash_atten':
        model = GPT2MolGen_flash_atten(**cfg.model)
    else:
        model = GPT2MolGen(**cfg.model)
    checkpoint = torch.load(os.path.join(output_dir, 'pytorch_model.bin'), map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model = model.to(dtype=torch.bfloat16, device='cuda')

    if cfg.wandb_logs:
        wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project,
                   name=exp_name, config=OmegaConf.to_container(cfg),
                   tags=cfg.wandb.tags, mode=cfg.wandb.mode)

    # Generate SMILES and calculate metrics
    generated_smiles = generate_smiles_HF(
        model,
        datamodule.tokenizer,
        cfg.eval.n_samples,
        cfg.eval.prompt,
        cfg.eval.temperature,
        cfg.eval.top_k,
        cfg.eval.top_p,
        cfg.eval.batch_size,
        device=torch.device('cuda')
    )
    metrics = get_all_metrics(generated_smiles, n_jobs=cfg.eval.preprocess_num_jobs)
    if cfg.wandb_logs:
        wandb.log(metrics)

    # Convert the dictionary to a list of lists
    metrics_table = [[k, v] for k, v in metrics.items()]
    print(tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="pretty"))

    # Convert the list of lists to a DataFrame
    df = pd.DataFrame(metrics_table, columns=["Metric", "Value"])
    df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)

    # Save the SMILES to a CSV file
    df = pd.DataFrame(generated_smiles, columns=["SMILES"])
    df.to_csv(os.path.join(output_dir, 'generated_smiles.csv'), index=False)


if __name__ == "__main__":
    entrypoint()
