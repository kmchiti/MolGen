import hydra
import os
import numpy as np
from omegaconf import DictConfig, OmegaConf
from dataset import MolGenDataModule
from datasets import load_dataset
from tqdm import tqdm

@hydra.main(version_base=None, config_path="configs/dataset", config_name="pubchem_smiles")
def entrypoint(cfg: DictConfig):

    # Initialize DataModule
    datamodule = MolGenDataModule(**cfg)
    # print(datamodule.tokenizer)
    # datamodule.setup()
    # print(datamodule.train_dataset)
    # print(datamodule.eval_dataset)

    dataset = load_dataset(datamodule.dataset_name, num_proc=datamodule.dataloader_num_workers)
    if 'test' not in dataset.keys():
        dataset = dataset["train"].train_test_split(test_size=datamodule.validation_size,
                                                    seed=datamodule.val_split_seed)

    def tokneize_and_extract_lengths(
            element,
            mol_type,
            tokenizer,
    ):
        outputs = tokenizer(
            element[mol_type],
            truncation=True,
            add_special_tokens=True,
        )

        return {"input_ids": outputs["input_ids"], "lengths": [len(x) for x in outputs['input_ids']]}

    # Map function to extract lengths
    lengths_dataset = dataset.map(
        tokneize_and_extract_lengths,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=datamodule.preprocess_num_workers,
        fn_kwargs={
            "mol_type": datamodule.mol_type,
            "tokenizer": datamodule.tokenizer,
        },
    )

    batch_size = 1000000
    total_rows = len(lengths_dataset['train'])

    max_len = 0
    for i in tqdm(range(0, total_rows, batch_size)):
        max_len = max(max_len, max(lengths_dataset['train']['lengths'][i:i + batch_size]))

    print('max_len:', max_len)


if __name__ == "__main__":
    datamodule = entrypoint()
