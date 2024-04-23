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

    def compute_histogram(data_chunk):
        return np.bincount(data_chunk, minlength=max_len+1)
    # Number of splits; generally equal to or a multiple of the number of cores
    num_splits = 24
    chunk_size = len(lengths_dataset['train']['lengths']) // num_splits

    histograms = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_splits) as executor:
        futures = [executor.submit(compute_histogram, lengths_dataset['train']['lengths'][i * chunk_size:(i + 1) * chunk_size]) for i in
                   range(num_splits)]
        for future in concurrent.futures.as_completed(futures):
            histograms.append(future.result())

    # Aggregate the histograms
    final_histogram = np.sum(histograms, axis=0)
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
