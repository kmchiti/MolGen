import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import set_seed, Trainer, TrainingArguments
from dataset import MolGenDataModule
from models import GPT2MolGen, GPT2MolGen_flash_atten
from utils import creat_unique_experiment_name, is_world_process_zero, save_HF_model
import wandb
import torch
import os


def _checkpoint_is_available(trainer):
    items = os.listdir(trainer.args.output_dir)
    checkpoint_found = any(
        item.startswith("checkpoint") and os.path.isdir(os.path.join(trainer.args.output_dir, item)) for item
        in items)
    return checkpoint_found

@hydra.main(version_base=None, config_path="configs", config_name="config")
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
    else:
        model = GPT2MolGen(**cfg.model)
    checkpoint = torch.load(os.path.join(output_dir, 'pytorch_model.bin'), map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)

    # Initialize trainer
    if cfg.wandb_logs:
        train_args = TrainingArguments(**cfg.trainer, output_dir=output_dir, data_seed=cfg.seed,
                                       seed=cfg.seed, logging_dir=output_dir, report_to=['wandb'])
        if is_world_process_zero(train_args):
            wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project,
                       name=exp_name, config=OmegaConf.to_container(cfg),
                       tags=cfg.wandb.tags, mode=cfg.wandb.mode)

    else:
        os.environ["WANDB_DISABLED"] = "true"
        train_args = TrainingArguments(**cfg.trainer, output_dir=output_dir, data_seed=cfg.seed,
                                       seed=cfg.seed, logging_dir=output_dir, report_to=None)

    trainer = Trainer(model=model,
                      args=train_args,
                      data_collator=datamodule.data_collator,
                      train_dataset=datamodule.train_dataset,
                      eval_dataset=datamodule.eval_dataset,)

    # Evaluate the model
    results = trainer.evaluate()
    # Print the accuracy or other relevant metrics
    print(results)

    if is_world_process_zero(train_args):
        print('save model to:', os.path.join(output_dir, 'HF'))
        save_HF_model(model, model.config, os.path.join(output_dir, 'HF'))


if __name__ == "__main__":
    entrypoint()
