import hydra
from omegaconf import DictConfig, OmegaConf
from dataset import MolGenDataModule


@hydra.main(version_base=None, config_path="configs/dataset", config_name="pubchem_smiles")
def entrypoint(cfg: DictConfig):

    # Initialize DataModule
    datamodule = MolGenDataModule(**cfg)
    print(datamodule.tokenizer)
    datamodule.setup()
    print(datamodule.train_dataset)
    print(datamodule.eval_dataset)


if __name__ == "__main__":
    datamodule = entrypoint()
