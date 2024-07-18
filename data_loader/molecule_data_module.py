import os
import warnings
from datasets import load_from_disk, load_dataset
from datasets.arrow_dataset import Dataset
from datasets.naming import camelcase_to_snakecase
from datasets.config import HF_CACHE_HOME
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    AutoTokenizer,
)

class MolDataModule(object):
    def __init__(
            self,
            dataset_name: str = None,
            tokenizer_path: str = None,
            tokenizer_name: str = None,
            mol_type: str = "SMILES",
            max_seq_length: int = 64,
            num_proc: int = 4,
            streaming: bool = False,
            fix_validation_name: str = "MolGen/ZINC_270M-raw",
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.max_seq_length = max_seq_length
        self.num_proc = num_proc
        self.mol_type = mol_type
        self.tokenizer_name = tokenizer_name
        self.tokenizer_path = tokenizer_path
        self.streaming = streaming

        _tok_name = tokenizer_name.replace('/', '_') if tokenizer_name is not None else \
            tokenizer_path.split("tokenizers/")[1].split(".json")[0]
        _dat_path = self.get_cache_dir(dataset_name)
        self.save_directory = os.path.join(HF_CACHE_HOME, 'datasets', _dat_path, f'tokenized_{_tok_name}')

        # Load tokenizer
        if tokenizer_name is not None:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        elif tokenizer_path is not None:
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        else:
            raise "tokenizer not found"

        tokenizer.eos_token = "<eos>"
        tokenizer.bos_token = "<bos>"
        tokenizer.pad_token = "[PAD]"
        self.tokenizer = tokenizer

        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )
        # We use fix validation set over all the dataset
        if fix_validation_name == "MolGen/ZINC_270M-raw":  # about 27M molecules
            self.eval_dataset = load_dataset(
                os.path.join(HF_CACHE_HOME, "datasets", "MolGen___zinc_270_m-raw/default/0.0.0/c7a45b7083c92b44"),
                num_proc=self.num_proc,
                split='validation')
        else:
            self.eval_dataset = load_dataset(fix_validation_name, split='valid', num_proc=num_proc)
        self.valid_set = set(self.eval_dataset["SMILES"])  # this set will use 1GB of RAM
        self.train_dataset = None

    @staticmethod
    def get_cache_dir(dataset_name):
        namespace_and_dataset_name = dataset_name.split("/")
        namespace_and_dataset_name[-1] = camelcase_to_snakecase(namespace_and_dataset_name[-1])
        cached_relative_path = "___".join(namespace_and_dataset_name)
        return cached_relative_path

    # TODO:  no idea why it's not working
    # @staticmethod
    # def filter_smiles(
    #         example: dict,
    #         mol_type: str,
    #         valid_set: Dataset,
    # ):
    #     # Returns False if the SMILES is in the validation set, True otherwise
    #     return example[mol_type] not in valid_set

    def filter_smiles(self, example: dict):
        # Returns False if the SMILES is in the validation set, True otherwise
        return example[self.mol_type] not in self.valid_set

    @staticmethod
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

    def creat_tokenized_datasets(self):
        assert self.streaming is False

        # Load dataset
        dataset = load_dataset(self.dataset_name, num_proc=self.num_proc, split="train")

        # Filter validation data
        # dataset = dataset.filter(
        #     self.filter_smiles,
        #     batched=True,
        #     num_proc=self.num_proc,
        #     fn_kwargs={
        #         "mol_type": self.mol_type,
        #         "valid_set": self.valid_set,
        #     },
        # )
        dataset = dataset.filter(self.filter_smiles, batched=True, num_proc=self.num_proc)

        # Tokenize dataset
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            num_proc=self.num_proc,
            fn_kwargs={
                "max_length": self.max_seq_length,
                "mol_type": self.mol_type,
                "tokenizer": self.tokenizer,
            },
        )

        tokenized_dataset.save_to_disk(self.save_directory)
        print(f'tokenized dataset saved at: {self.save_directory}')

    def prepare_tokenized_streaming_dataset(self):

        dataset = load_dataset(self.dataset_name, split="train", streaming=True)
        column_names = list(dataset.features)
        # dataset = dataset.filter(
        #     self.filter_smiles,
        #     fn_kwargs={
        #         "mol_type": self.mol_type,
        #         "valid_set": self.valid_set,
        #     }
        # )
        dataset = dataset.filter(self.filter_smiles)

        tokenized_dataset = dataset.map(
            self.tokenize_function,
            remove_columns=column_names,
            fn_kwargs={
                "max_length": self.max_seq_length,
                "mol_type": self.mol_type,
                "tokenizer": self.tokenizer,
            },
        )
        print(f'prepare streaming dataset')
        return tokenized_dataset

    def load_tokenized_dataset(self):
        if self.streaming:
            self.train_dataset = self.prepare_tokenized_streaming_dataset()

        else:
            if not os.path.exists(self.save_directory):
                warnings.warn("tokenized dataset didn't found locally.\ncreating tokenized dataset may takes time.")
                self.creat_tokenized_datasets()

            tokenized_dataset = load_from_disk(self.save_directory)
            self.train_dataset = tokenized_dataset


