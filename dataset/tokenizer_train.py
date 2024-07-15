from datasets import load_dataset
from typing import List

from tokenizers import Tokenizer, Regex, NormalizedString, PreTokenizedString
from tokenizers.processors import TemplateProcessing
from tokenizers.pre_tokenizers import Split, PreTokenizer, Whitespace
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
import argparse
import os


def atomwise_tokenizer(smi: str, exclusive_tokens=None) -> List[str]:
    """Tokenize a SMILES molecule at atom-level:
        (1) 'Br' and 'Cl' are two-character tokens
        (2) Symbols with bracket are considered as tokens

    :param smi: A SMILES string
    :param exclusive_tokens: A list of specifical symbols with bracket you want to keep. e.g., ['[C@@H]', '[nH]'], defaults to None
    :return: A list of atom-wise tokens
    """
    import re

    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"  # type: ignore
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]

    if exclusive_tokens:
        for i, tok in enumerate(tokens):
            if tok.startswith("["):
                if tok not in exclusive_tokens:
                    tokens[i] = "[UNK]"
    return tokens


def tokens_to_mer(toks):
    return "".join(toks)


def kmer_tokenizer(
    smiles: str,
    ngram: int = 4,
    stride: int = 1,
    remove_last: bool = False,
    exclusive_tokens=None,
) -> List[str]:
    """Tokenize a SMILES molecule at k-mer level.

    :param smiles: A SMILES string
    :param ngram: kmer size, defaults to 4
    :param stride: kmer stride, defaults to 1
    :param remove_last: Whether to remove last token, defaults to False
    :param exclusive_tokens: A list of specifical symbols with bracket you want to keep. e.g., ['[C@@H]', '[nH]'], defaults to None
    :return: A list of k-mer tokens
    """
    units = atomwise_tokenizer(
        smiles, exclusive_tokens=exclusive_tokens
    )  # collect all the atom-wise tokens from the SMILES
    if ngram == 1:
        tokens = units
    else:
        tokens = [
            tokens_to_mer(units[i : i + ngram])
            for i in range(0, len(units), stride)
            if len(units[i : i + ngram]) == ngram
        ]

    if remove_last:
        if (
            len(tokens[-1]) < ngram
        ):  # truncate last whole k-mer if the length of the last k-mers is less than ngram.
            tokens = tokens[:-1]
    return tokens


class KmerPreTokenizer:
    def __init__(
        self,
        ngram: int = 4,
        stride: int = 1,
        remove_last: bool = False,
        exclusive_tokens=None,
    ):
        self.ngram = ngram
        self.stride = stride
        self.remove_last = remove_last
        self.exclusive_tokens = exclusive_tokens

    def kmer_split(
        self, i: int, normalized_string: NormalizedString
    ) -> List[NormalizedString]:
        splits = []
        # we need to call `str(normalized_string)` because kmer expects a str,
        # not a NormalizedString
        for token in kmer_tokenizer(
            str(normalized_string),
            ngram=self.ngram,
            stride=self.stride,
            remove_last=self.remove_last,
            exclusive_tokens=self.exclusive_tokens,
        ):
            splits.append(NormalizedString(token))  # type: ignore

        return splits

    def pre_tokenize(self, pretok: PreTokenizedString):
        # Let's call split on the PreTokenizedString to split using `self.kmer_split`
        pretok.split(self.kmer_split)


def train_tokenizer(
    dataset_name: str = "PubChem",
    mol_type: str = "SMILES",
    tokenizer_method: str = "atomwise",
    ngram: int = 4,
    stride: int = 1,
    remove_last: bool = False,
    exclusive_tokens=None,
    max_vocab_size: int = 30000,
    path_to_save: str = "data/processed/training/tokenizers",
) -> None:
    data = load_dataset(f"MolGen/{dataset_name}", split='train')
    smiles = data[mol_type]  # type: ignore

    # Define a tokenizer
    tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))  # type: ignore

    if tokenizer_method == "atomwise":
        # Add pre-tokenization and decoding processes
        tokenizer.pre_tokenizer = Split(
            Regex(
                r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
            ),
            behavior="isolated",
        )  # type: ignore
    elif tokenizer_method == "kmer":
        tokenizer.pre_tokenizer = PreTokenizer.custom(  # type: ignore
            KmerPreTokenizer(
                ngram=ngram,
                stride=stride,
                remove_last=remove_last,
                exclusive_tokens=exclusive_tokens,
            )
        )
    else:
        raise ValueError("Invalid tokenizer method")

    # Train the tokenizer on your sequences
    trainer = WordLevelTrainer(
        vocab_size=max_vocab_size, special_tokens=["<unk>", "<bos>", "<eos>", "[PAD]"]
    )  # type: ignore
    tokenizer.train_from_iterator(smiles, trainer=trainer)

    # Add post-processing
    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[
            ("<bos>", 1),
            ("<eos>", 2),
        ],
    )  # type: ignore

    # Save the tokenizer
    if tokenizer_method == "kmer":
        tokenizer.pre_tokenizer = Whitespace()  # type: ignore
    tokenizer.save(os.path.join(path_to_save, f"{tokenizer_method}_{dataset_name}_{mol_type}_{max_vocab_size}.json"))


if __name__ == "__main__":
    print('===========start===========')
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="ChEMBL")
    parser.add_argument("--mol_type", type=str, default="SMILES")
    parser.add_argument("--tokenizer_method", type=str, default="atomwise")
    parser.add_argument("--max_vocab_size", type=int, default=30000)
    args = parser.parse_args()
    train_tokenizer(
        args.dataset_name, args.mol_type, args.tokenizer_method, args.max_vocab_size
    )
