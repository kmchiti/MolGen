import hydra
from omegaconf import DictConfig, OmegaConf
from data_loader import MolDataModule
from torch.utils.data import DataLoader


# mode = 'tokenize'
mode = 'test_streaming'

@hydra.main(version_base=None, config_path="configs/dataset", config_name="ZINC_270M_smiles_atomwise")
def entrypoint(cfg: DictConfig):
    if mode == 'tokenize':
        # Initialize DataModule
        datamodule = MolDataModule(**cfg)
        print(datamodule.tokenizer)
        datamodule.creat_tokenized_datasets()

        print(datamodule.train_dataset)
        print(datamodule.eval_dataset)

    elif mode == 'test_streaming':
        # Initialize DataModule
        datamodule = MolDataModule(**cfg)
        datamodule.load_tokenized_dataset()

        dataloader_params = {
            "batch_size": 8,
            "collate_fn": datamodule.data_collator,
            "num_workers": datamodule.num_proc,
            "pin_memory": True,
        }
        train_data_loader = DataLoader(datamodule.train_dataset, **dataloader_params)
        eval_data_loader = DataLoader(datamodule.eval_dataset, **dataloader_params)

        for i, batch in enumerate(train_data_loader):
            print(i, batch['input_ids'].shape)
            if i > 10:
                break
        for i, batch in enumerate(eval_data_loader):
            print(i, batch['input_ids'].shape)
            if i > 10:
                break


if __name__ == "__main__":
    datamodule = entrypoint()
