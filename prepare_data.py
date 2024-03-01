import hydra
from omegaconf import DictConfig, OmegaConf
from dataset import MolGenDataModule


@hydra.main(version_base=None, config_path="configs/dataset", config_name="pubchem_smiles")
def entrypoint(cfg: DictConfig):

    # Initialize DataModule
    datamodule = MolGenDataModule(**cfg.dataset)
    datamodule.setup()
    print(datamodule.tokenizer)
    print(datamodule.train_dataset)
    print(datamodule.eval_dataset)


if __name__ == "__main__":
    entrypoint()
