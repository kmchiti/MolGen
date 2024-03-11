import hydra
from omegaconf import DictConfig, OmegaConf
from dataset import MolGenDataModule


@hydra.main(version_base=None, config_path="configs/dataset", config_name="pubchem_smiles")
def entrypoint(cfg: DictConfig):

    # Initialize DataModule
    datamodule = MolGenDataModule(**cfg)
    return datamodule
    # print(f"maximum sequence length in samples: {datamodule.get_maximum_length()}")
    # datamodule.setup()
    # print(datamodule.tokenizer)
    # print(datamodule.train_dataset)
    # print(datamodule.eval_dataset)


import multiprocessing
from functools import partial
from datasets import load_dataset

def tokenize_and_get_length(sequence, tokenizer):
    tokens = tokenizer(sequence, truncation=True, padding=True, return_tensors='pt')['input_ids']
    return tokens.size(1)


if __name__ == "__main__":
    datamodule = entrypoint()
    # Load dataset
    dataset = load_dataset(datamodule.dataset_name, num_proc=datamodule.dataloader_num_workers)
    sequences = dataset["train"][datamodule.mol_type]
    pool = multiprocessing.Pool(processes=datamodule.preprocess_num_workers)
    tokenize_partial = partial(tokenize_and_get_length, tokenizer=datamodule.tokenizer)
    token_lengths = pool.map(tokenize_partial, sequences)
    pool.close()
    pool.join()
    print("maximum sequence length in samples", max(token_lengths))

