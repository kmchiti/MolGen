from eval import get_all_metrics, generate_smiles_HF, generate_smiles_FA
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import set_seed, Trainer, TrainingArguments
from trainer import MyHFTrainer, MyTrainingArguments
from callbacks import WandbCallback
from dataset import MolGenDataModule
from models import GPT2MolGen, GPT2MolGen_flash_atten, Llama_small_flash_atten
from utils import creat_unique_experiment_name, is_world_process_zero, save_HF_model
import torch
from transformers import LlamaForCausalLM, LlamaConfig
from flash_attn.models.gpt import GPTLMHeadModel
from tabulate import tabulate
import os


@hydra.main(version_base=None, config_path="configs", config_name="config_moses_finetune")
def entrypoint(cfg: DictConfig):
    # Initialize setup
    set_seed(cfg.seed)
    exp_name = creat_unique_experiment_name(cfg)
    output_dir = os.path.join(cfg.save_path, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize DataModule
    datamodule = MolGenDataModule(**cfg.dataset)
    datamodule.setup()

    # Load model
    if cfg.model.model_name_or_path == 'gpt2_flash_atten':
        model = GPT2MolGen_flash_atten(**cfg.model)
    elif cfg.model.model_name_or_path in ['llama_small', 'llama_small_HF']:
        model_cfg = LlamaConfig(**cfg.model)
        model = LlamaForCausalLM(model_cfg)
    elif cfg.model.model_name_or_path == 'llama_small_FA':
        model = Llama_small_flash_atten(**cfg.model)
    else:
        model = GPT2MolGen(**cfg.model)

    # Initialize trainer
    train_args = MyTrainingArguments(data_seed=cfg.seed, seed=cfg.seed, output_dir=output_dir,
                                     **cfg.trainer)
    # enable TF32
    if torch.cuda.is_available() and cfg.trainer.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if cfg.wandb_logs:
        wandb_callback = [WandbCallback(model=model, entity=cfg.wandb.entity, project=cfg.wandb.project,
                                        name=exp_name, config=OmegaConf.to_container(cfg), tags=cfg.wandb.tags)]
    else:
        wandb_callback = None

    trainer = MyHFTrainer(model=model,
                          args=train_args,
                          callbacks=wandb_callback,
                          tokenizer=datamodule.tokenizer,
                          data_collator=datamodule.data_collator,
                          train_dataset=datamodule.train_dataset,
                          eval_dataset=datamodule.eval_dataset,
                          evaluation_task=None,
                          )

    # Load checkpoint
    checkpoint = os.path.join(cfg.save_path, cfg.model_name_or_path)  # "llama_small_FA_PubChem"
    print(f"load checkpoint from: {checkpoint}")
    trainer._load_from_checkpoint(checkpoint)

    # Train model
    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload
    if is_world_process_zero(train_args) and isinstance(model, GPTLMHeadModel):
        print('save remapped HF model to:', os.path.join(output_dir, 'HF'))
        model.save_HF_model(OmegaConf.to_container(cfg.model), datamodule.tokenizer,
                            output_dir=os.path.join(output_dir, 'HF'),
                            dataset_name=cfg.dataset['dataset_name'].replace("MolGen/", ""),
                            token=cfg.write_HF_token)


if __name__ == "__main__":
    entrypoint()
