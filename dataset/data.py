import warnings
import os
import psutil
from datasets import Features, Value, load_from_disk, load_dataset
from dataset import my_load_dataset, get_cache_dir
from typing import Optional
from torch.utils.data.dataloader import DataLoader
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    AutoTokenizer,
)
from tokenizers.processors import TemplateProcessing

class MolGenDataModule(object):
    def __init__(
        self,
        dataset_name: str = None,
        tokenizer_path: str = None,
        tokenizer_name: str = None,
        mol_type: str = "SMILES",
        overwrite_cache: bool = None,
        max_seq_length: int = 64,
        dataloader_num_workers: int = 4,
        preprocess_num_workers: int = 34,
        validation_size: Optional[float] = 0.1,
        val_split_seed: Optional[int] = 42,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.overwrite_cache = overwrite_cache
        self.max_seq_length = max_seq_length
        self.dataloader_num_workers = dataloader_num_workers
        self.preprocess_num_workers = preprocess_num_workers
        self.validation_size = validation_size
        self.val_split_seed = val_split_seed
        self.mol_type = mol_type
        self.tokenizer_name = tokenizer_name
        self.tokenizer_path = tokenizer_path

        self.data_collator = None
        self.eval_dataset = None
        self.train_dataset = None

        # Load tokenizer
        if tokenizer_name is not None:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        elif tokenizer_path is not None:
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        else:
            raise "tokenizer not found"

        tokenizer.eos_token = "<eos>"
        tokenizer.bos_token = "<bos>"
        if tokenizer_path is not None and "ZINC_250k_SMILES_bpe_30000" in tokenizer_path:
            tokenizer.pad_token = "<pad>"
        else:
            tokenizer.pad_token = "[PAD]"

        if tokenizer_path is not None and "BPE_pubchem_500" in tokenizer_path:
            tokenizer.add_special_tokens({"additional_special_tokens": ["<bos>", "<eos>", "[PAD]"]})  # type: ignore
            # Add special tokens to the post processor
            tokenizer._tokenizer.post_processor = TemplateProcessing(
                single="<bos> $A <eos>",
                special_tokens=[
                    ("<bos>", tokenizer.bos_token_id),
                    ("<eos>", tokenizer.eos_token_id),
                ],
            )

        self.tokenizer = tokenizer
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )
        _tok_name = tokenizer_name if tokenizer_name is not None else tokenizer_path.split("tokenizers/")[1].split(".json")[0]
        _dat_path = get_cache_dir(dataset_name)
        self.save_directory = os.path.join(os.environ["HF_HOME"],
                                           f'datasets/{_dat_path}/tokenized_{_tok_name}')
        print(self.save_directory)

    def creat_tokenized_datasets(self):

        # Load dataset
        # if self.dataset_name == 'MolGen/ZINC_22-raw':
        #     print('************ load from file ************')
        #     _dat_path = "MolGen___" + self.dataset_name.split("MolGen/")[1].lower()
        #     dataset = my_load_dataset(path=os.path.join(os.environ["HF_HOME"], f'datasets/{_dat_path}'), split="train",
        #                            num_proc=self.dataloader_num_workers)
        # else:
        #     dataset = my_load_dataset(self.dataset_name, num_proc=self.dataloader_num_workers)
        print('************ load from file ************', f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
        dataset = load_dataset(self.dataset_name, num_proc=self.dataloader_num_workers)
        print('************ load done! ************', f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

        if 'test' not in dataset.keys():
            print('************ start splitting ************')
            dataset = dataset["train"].train_test_split(test_size=self.validation_size, seed=self.val_split_seed)
            print('************ splitting done! ************', f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

        def tokenize_function(
            element: dict,
            max_length: int,
            mol_type: str,
            tokenizer: PreTrainedTokenizerFast,
        ) -> dict:
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

        print('************ start tokenizing ************')
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            num_proc=self.preprocess_num_workers,
            load_from_cache_file=not self.overwrite_cache,
            fn_kwargs={
                "max_length": self.max_seq_length,
                "mol_type": self.mol_type,
                "tokenizer": self.tokenizer,
            },
        )
        print('************ save to disk ************',  f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
        tokenized_dataset.save_to_disk(self.save_directory)
        print('************************')
        print(f'tokenized dataset saved at: {self.save_directory}')
        print('************************', f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

        # # Create train and validation datasets

    def train_dataloader(self, batch_size: int = 1024,):
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            collate_fn=self.data_collator,
            num_workers=self.dataloader_num_workers,
            shuffle=True,
        )

    def val_dataloader(self, batch_size: int = 1024,):
        return DataLoader(
            self.eval_dataset,
            batch_size=batch_size,
            collate_fn=self.data_collator,
            num_workers=self.dataloader_num_workers,
        )

    def load_tokenized_dataset(self):
        tokenized_dataset = load_from_disk(self.save_directory)
        self.train_dataset = tokenized_dataset["train"]
        self.eval_dataset = tokenized_dataset["test"]

    def get_maximum_length(self):
        import multiprocessing
        from functools import partial

        # Load dataset
        dataset = my_load_dataset(self.dataset_name, num_proc=self.dataloader_num_workers)
        sequences = dataset["train"][self.mol_type]
        def tokenize_and_get_length(sequence):
            tokens = self.tokenizer(sequence, truncation=True, padding=True, return_tensors='pt')['input_ids']
            return tokens.size(1)

        pool = multiprocessing.Pool(processes=self.preprocess_num_workers)
        tokenize_partial = partial(tokenize_and_get_length)
        token_lengths = pool.map(tokenize_partial, sequences)
        pool.close()
        pool.join()
        return max(token_lengths)
