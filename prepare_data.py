import hydra
import os
import numpy as np
from omegaconf import DictConfig, OmegaConf
import concurrent.futures
from dataset import MolGenDataModule
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt


def set_plot_style(
        fsize: int = 14,
        tsize: int = 10,
        tdir: str = 'in',
        major: float = 1.0,
        minor: float = 1.0,
        style: str = 'default',
        use_latex_format: bool = False,
        linewidth: float = 2.0,
):
    plt.style.use(style)
    plt.rcParams['text.usetex'] = use_latex_format
    plt.rcParams['font.size'] = fsize
    plt.rcParams['legend.fontsize'] = tsize
    plt.rcParams['xtick.direction'] = tdir
    plt.rcParams['ytick.direction'] = tdir
    plt.rcParams['xtick.major.size'] = major
    plt.rcParams['xtick.minor.size'] = minor
    plt.rcParams['ytick.major.size'] = major
    plt.rcParams['ytick.minor.size'] = minor
    plt.rcParams['lines.linewidth'] = linewidth


tokenize = False


@hydra.main(version_base=None, config_path="configs/dataset", config_name="pubchem_smiles")
def entrypoint(cfg: DictConfig):
    if tokenize:
        # Initialize DataModule
        datamodule = MolGenDataModule(**cfg)
        print(datamodule.tokenizer)
        datamodule.setup()
        print(datamodule.train_dataset)
        print(datamodule.eval_dataset)

    else:
        datamodule = MolGenDataModule(**cfg)
        print(datamodule.tokenizer)
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

        batch_size = 5000000
        total_rows = len(lengths_dataset['train'])

        max_len = 0
        for i in tqdm(range(0, total_rows, batch_size)):
            max_temp = np.max(lengths_dataset['train']['lengths'][i:i + batch_size])
            max_len = max(max_len, max_temp)

        # total_rows = len(lengths_dataset['test'])
        # for i in tqdm(range(0, total_rows, batch_size)):
        #     max_temp = np.max(lengths_dataset['test']['lengths'][i:i + batch_size])
        #     max_len = max(max_len, max_temp)

        print('max_len:', max_len)

        histogram = []
        for i in tqdm(range(0, total_rows, batch_size)):
            hist = np.bincount(lengths_dataset['train']['lengths'][i:i + batch_size], minlength=max_len + 1)
            histogram.append(hist)

        # Aggregate the histograms
        final_histogram = np.sum(histogram, axis=0)
        if datamodule.tokenizer_path is not None:
            save_path = datamodule.tokenizer_path.split('.json')[0] + '.npy'
            np.save(save_path, final_histogram)
        else:
            save_path = './data/processed/training/tokenizers' + f'{datamodule.tokenizer_name}.npy'
            np.save(save_path, final_histogram)

        set_plot_style()
        fig, axes = plt.subplots(figsize=(8, 5))
        fig.tight_layout()
        train_counts = final_histogram
        train_bins = np.arange(len(final_histogram))
        axes.hist(train_bins, weights=train_counts, log=True, alpha=0.7)
        axes.legend()
        axes.set_xlabel('Length of samples')
        axes.grid()
        plt.savefig(f'{save_path[:-4]}.png')


if __name__ == "__main__":
    datamodule = entrypoint()
