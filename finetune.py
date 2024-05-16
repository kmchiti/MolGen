from hydra import compose, initialize
from trainer import MyHFTrainer, MyTrainingArguments
from callbacks import WandbCallback
from dataset import MolGenDataModule
from models import GPT2MolGen, GPT2MolGen_flash_atten, Llama_small_flash_atten
from utils import creat_unique_experiment_name, is_world_process_zero, save_HF_model
from omegaconf import DictConfig, OmegaConf
import torch
import wandb
import argparse
from transformers import LlamaForCausalLM, LlamaConfig
from flash_attn.models.gpt import GPTLMHeadModel
from tabulate import tabulate
import pandas as pd
from eval import generate_smiles_HF, generate_smiles_FA, get_all_metrics
import os
from datasets import Dataset, DatasetDict
from pandarallel import pandarallel
from generated_and_filter import get_qed, get_sa, get_logp

def args_parser():
    parser = argparse.ArgumentParser(
        description='Molecules Generation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--exp_name', default="llama_small_FA_ZINC_270M_3d67f607_", type=str,
                        help='name of the trained config')
    parser.add_argument('--config_name', default="finetune_ZINC_270M_atomwise", type=str,
                        help='name of the trained config')
    parser.add_argument('--task', default='qed', type=str, help='task to filter for')
    args = parser.parse_args()
    return args


def entrypoint(args):
    # Initialize setup
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name=args.config_name)

    # Initialize setup
    exp_name = args.exp_name
    output_dir = os.path.join(cfg.save_path, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize DataModule
    datamodule = MolGenDataModule(**cfg.dataset)
    datamodule.setup()
    df = pd.read_csv(f'{output_dir}/generated_smiles_qed_42.csv')
    dataset = Dataset.from_dict(df)
    dataset = DatasetDict({"train": dataset})

    def tokenize_function(
            element: dict,
            max_length: int,
            mol_type: str,
            tokenizer,
    ):
        """Tokenize a single element of the dataset.

        Args:
            element (dict): Dictionary with the data to be tokenized.
            max_length (int): Maximum length of the tokenized sequence.
            mol_type (str): mol_type of the dataset to be tokenized.
            tokenizer (PreTrainedTokenizerFast): Tokenizer to be used.

        Returns:
            dict: Dictionary with the tokenized data.
        """
        outputs = tokenizer(
            element[mol_type],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            add_special_tokens=True,
        )
        return {"input_ids": outputs["input_ids"]}

    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=datamodule.preprocess_num_workers,
        load_from_cache_file=not datamodule.overwrite_cache,
        fn_kwargs={
            "max_length": datamodule.max_seq_length,
            "mol_type": datamodule.mol_type,
            "tokenizer": datamodule.tokenizer,
        },
    )

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
    if cfg.wandb_logs:
        wandb_callback = [WandbCallback(model=model, entity=cfg.wandb.entity, project=cfg.wandb.project,
                                        name=f"{exp_name}_finetune", config=OmegaConf.to_container(cfg), tags=cfg.wandb.tags)]
    else:
        wandb_callback = None
    output_dir_ = f"{output_dir}_finetune"
    train_args = MyTrainingArguments(data_seed=cfg.seed, seed=cfg.seed, output_dir=output_dir_, **cfg.trainer)

    trainer = MyHFTrainer(model=model,
                          args=train_args,
                          callbacks=wandb_callback,
                          tokenizer=datamodule.tokenizer,
                          data_collator=datamodule.data_collator,
                          train_dataset=tokenized_dataset["train"],
                          evaluation_task=None,
                          )

    print(f"load checkpoint from: {output_dir}")
    trainer._load_from_checkpoint(output_dir)
    train_result = trainer.train()
    trainer.save_model()

    # Generate SMILES and calculate metrics
    if isinstance(model, Llama_small_flash_atten) or isinstance(model, GPT2MolGen_flash_atten):
        generated_smiles = generate_smiles_FA(
            model=model,
            tokenizer=datamodule.tokenizer,
            n_samples=cfg.eval.n_samples,
            num_return_sequences=cfg.eval.batch_size,
            prompt=cfg.eval.prompt,
            temperature=cfg.eval.temperature,
            top_k=cfg.eval.top_k,
            top_p=cfg.eval.top_p,
            device=torch.device('cuda')
        )
    else:
        generated_smiles = generate_smiles_HF(
            model=model,
            tokenizer=datamodule.tokenizer,
            n_samples=cfg.eval.n_samples,
            num_return_sequences=cfg.eval.batch_size,
            prompt=cfg.eval.prompt,
            temperature=cfg.eval.temperature,
            top_k=cfg.eval.top_k,
            top_p=cfg.eval.top_p,
            device=torch.device('cuda')
        )

    # Save the SMILES to a CSV file
    df = pd.DataFrame(generated_smiles, columns=["SMILES"])
    pandarallel.initialize(shm_size_mb=60720, nb_workers=args.preprocess_num_jobs, progress_bar=True)
    df['qed'] = df['SMILES'].parallel_apply(get_qed)
    df.to_csv(os.path.join(output_dir_, 'generated_smiles.csv'), index=False)


if __name__ == "__main__":
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    args = args_parser()
    entrypoint(args)
