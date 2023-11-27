import warnings
import os
from datasets import Features, Value, load_dataset
from typing import Optional
from torch.utils.data.dataloader import DataLoader
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
)


class MolGenDataModule(object):
    def __init__(
        self,
        data_root: str,
        tokenizer_path: str,
        dataset_path: str,
        file_type: str,
        overwrite_cache: bool,
        max_seq_length: int,
        batch_size: int,
        dataloader_num_workers: int,
        preprocess_num_workers: int,
        folder_url: Optional[str] = None,
    ):
        super().__init__()
        self.tokenizer_path = os.path.join(data_root, tokenizer_path)
        self.dataset_path = os.path.join(data_root, dataset_path)
        self.file_type = file_type
        self.overwrite_cache = overwrite_cache
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.preprocess_num_workers = preprocess_num_workers
        self.folder_url = folder_url
        self.data_collator = None
        self.eval_dataset = None
        self.train_dataset = None

    def _dataset_available(self):
        return os.path.exists(self.tokenizer_path) and os.path.exists(self.dataset_path[:-1])

    def download_dataset(self):
        import gdown

        folder_id = self.folder_url.split('/')[-1]
        gdown.cached_download(f"https://drive.google.com/uc?id={folder_id}", f"{folder_id}.json")

        # Extract file IDs from the metadata
        with open(f"{folder_id}.json", 'r') as f:
            metadata = f.read()

        file_ids = [line.split('"')[3] for line in metadata.split('\n') if '"id"' in line]

        # Download each file in the folder
        for file_id in file_ids:
            gdown.download(f"https://drive.google.com/uc?id={file_id}", output=f"{file_id}.txt")

    def setup(self):

        if not self._dataset_available():
            warnings.warn('dataset is not available! download the dataset')
            self.download_dataset()

        # Load tokenizer
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.tokenizer_path)
        tokenizer.add_special_tokens({"additional_special_tokens": ["<bos>", "<eos>"]})  # type: ignore
        tokenizer.eos_token = "<eos>"
        tokenizer.bos_token = "<bos>"
        tokenizer.pad_token = tokenizer.eos_token

        # Load dataset
        dataset = load_dataset(
            self.file_type,
            data_files=self.dataset_path,
            features=Features(
                {"CID": Value(dtype="string"), "SMILES": Value(dtype="string")}
            ),
        )
        dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)  # type: ignore

        def tokenize_function(
            element: dict,
            max_length: int,
            column: str,
            tokenizer: PreTrainedTokenizerFast,
        ) -> dict:
            """Tokenize a single element of the dataset.

            Args:
                element (dict): Dictionary with the data to be tokenized.
                max_length (int): Maximum length of the tokenized sequence.
                column (str): Column of the dataset to be tokenized.
                tokenizer (PreTrainedTokenizerFast): Tokenizer to be used.

            Returns:
                dict: Dictionary with the tokenized data.
            """
            outputs = tokenizer(
                element[column],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            return {"input_ids": outputs["input_ids"]}

        # Tokenize dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            num_proc=self.preprocess_num_workers,
            load_from_cache_file=not self.overwrite_cache,
            fn_kwargs={
                "max_length": self.max_seq_length,
                "column": "SMILES",
                "tokenizer": tokenizer,
            },
        )

        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        )

        # Create train and validation datasets
        train_dataset = tokenized_dataset["train"]
        eval_dataset = tokenized_dataset["test"]
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,  # type: ignore
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            num_workers=self.dataloader_num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_dataset,  # type: ignore
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            num_workers=self.dataloader_num_workers,
        )
