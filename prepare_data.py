import hydra
from omegaconf import DictConfig, OmegaConf
from data_loader import MolDataModule


mode = 'tokenize'

@hydra.main(version_base=None, config_path="configs/dataset", config_name="pubchem_smiles")
def entrypoint(cfg: DictConfig):
    if mode == 'tokenize':
        # Initialize DataModule
        datamodule = MolDataModule(**cfg)
        print(datamodule.tokenizer)
        datamodule.creat_tokenized_datasets()

        print(datamodule.train_dataset)
        print(datamodule.eval_dataset)


if __name__ == "__main__":
    datamodule = entrypoint()
