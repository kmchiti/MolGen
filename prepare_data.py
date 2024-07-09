import hydra
import os
import numpy as np
from omegaconf import DictConfig, OmegaConf
import concurrent.futures
from dataset import MolGenDataModule
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold

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


mode = 'tokenize'


@hydra.main(version_base=None, config_path="configs/dataset", config_name="pubchem_smiles")
def entrypoint(cfg: DictConfig):

    if mode == 'tokenize':
        # Initialize DataModule
        datamodule = MolGenDataModule(**cfg)
        print(datamodule.tokenizer)
        # datamodule.creat_tokenized_datasets()

        # print(datamodule.train_dataset)
        # print(datamodule.eval_dataset)

    elif mode == 'scaffold':
        # def compute_scaffold(smi):
        #     try:
        #         mol = Chem.MolFromSmiles(smi)
        #         scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        #         return scaffold
        #     except:
        #         return None
        #
        # def process_batch(smiles_list):
        #     scaffold_set = set()
        #     for smi in smiles_list:
        #         scaffold = compute_scaffold(smi)
        #         if scaffold is not None:
        #             scaffold_set.add(scaffold)
        #     return scaffold_set
        #
        # def parallel_scaffold_extraction(dataset_split, num_workers=8):
        #     # Split data into manageable chunks
        #     chunk_size = len(dataset_split) // num_workers + 1  # plus one to handle rounding
        #     chunks = [dataset_split[i:i + chunk_size] for i in range(0, len(dataset_split), chunk_size)]
        #
        #     unique_scaffolds = set()
        #     with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        #         results = executor.map(process_batch, chunks)
        #         for result in results:
        #             unique_scaffolds.update(result)
        #     return unique_scaffolds

        # print('load dataset')
        # raw_dataset = load_dataset(cfg.dataset_name)
        # print('=========================compute scaffolds test set=========================')
        # test_scaffolds = parallel_scaffold_extraction(raw_dataset['test']['SMILES'])
        # print(test_scaffolds)
        # print('=========================compute scaffolds train set=========================')
        # train_scaffolds = parallel_scaffold_extraction(raw_dataset['train']['SMILES'])
        # print(train_scaffolds)
        # print('=========================compute scaffolds valid set=========================')
        # valid_scaffolds = parallel_scaffold_extraction(raw_dataset['valid']['SMILES'])
        # print(valid_scaffolds)
        # result = {'train_scaffolds': train_scaffolds, 'valid_scaffolds': valid_scaffolds, 'test_scaffolds': test_scaffolds}
        # result = pd.DataFrame(result)
        # save_path = f"./{cfg.dataset_name.split('/')[-1]}.csv"
        # result.to_csv(save_path)
        # print(f"save result in: {save_path}")

        print('load dataset')
        datamodule = MolGenDataModule(**cfg)
        print(datamodule.tokenizer)
        dataset = load_dataset(datamodule.dataset_name, num_proc=datamodule.dataloader_num_workers)

        from rdkit import Chem
        def compute_scaffold(smilse):
            if isinstance(smilse, list):
                scaffolds = []
                for smi in smilse:
                    scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(
                        mol=Chem.MolFromSmiles(smi), includeChirality=False
                    ))
                return scaffolds
            else:
                scaffolds = MurckoScaffold.MurckoScaffoldSmiles(
                    mol=Chem.MolFromSmiles(smilse), includeChirality=False
                )
                return scaffolds

        def extract_scaffolds(
                element,
                mol_type,
        ):

            return {"scaffold": compute_scaffold(element[mol_type])}


        batch_size = 5000000
        print('=========================compute scaffolds test set=========================')
        # Map function to extract lengths
        scaffolds_dataset = dataset['test'].map(
            extract_scaffolds,
            batched=True,
            remove_columns=dataset["train"].column_names,
            num_proc=datamodule.preprocess_num_workers,
            fn_kwargs={
                "mol_type": datamodule.mol_type,
            },
        )
        total_rows = len(scaffolds_dataset)
        test_scaffolds = set()
        for i in tqdm(range(0, total_rows, batch_size)):
            test_scaffolds.update(scaffolds_dataset['scaffold'][i:i + batch_size])
        print(f'# scaffolds in test set: {len(test_scaffolds)}')
        result = pd.DataFrame({'test_scaffolds': list(test_scaffolds)})
        save_path = f"./test_scaffolds_{cfg.dataset_name.split('/')[-1]}.csv"
        result.to_csv(save_path)
        print(f"save result in: {save_path}")

        print('=========================compute scaffolds valid set=========================')
        # Map function to extract lengths
        scaffolds_dataset = dataset['valid'].map(
            extract_scaffolds,
            batched=True,
            remove_columns=dataset["train"].column_names,
            num_proc=datamodule.preprocess_num_workers,
            fn_kwargs={
                "mol_type": datamodule.mol_type,
            },
        )
        total_rows = len(scaffolds_dataset)
        valid_scaffolds = set()
        for i in tqdm(range(0, total_rows, batch_size)):
            valid_scaffolds.update(scaffolds_dataset['scaffold'][i:i + batch_size])
        print(f'# scaffolds in valid set: {len(valid_scaffolds)}')
        result = pd.DataFrame({'valid_scaffolds': list(valid_scaffolds)})
        save_path = f"./valid_scaffolds_{cfg.dataset_name.split('/')[-1]}.csv"
        result.to_csv(save_path)
        print(f"save result in: {save_path}")

        print('=========================compute scaffolds train set=========================')
        # Map function to extract lengths
        scaffolds_dataset = dataset['train'].map(
            extract_scaffolds,
            batched=True,
            remove_columns=dataset["train"].column_names,
            num_proc=datamodule.preprocess_num_workers,
            fn_kwargs={
                "mol_type": datamodule.mol_type,
            },
        )
        total_rows = len(scaffolds_dataset)
        train_scaffolds = set()
        for i in tqdm(range(0, total_rows, batch_size)):
            train_scaffolds.update(scaffolds_dataset['scaffold'][i:i + batch_size])
        print(f'# scaffolds in train set: {len(train_scaffolds)}')
        result = pd.DataFrame({'train_scaffolds': list(train_scaffolds)})
        save_path = f"./train_scaffolds_{cfg.dataset_name.split('/')[-1]}.csv"
        result.to_csv(save_path)
        print(f"save result in: {save_path}")

    elif mode == 'substructures':
        from rdkit import Chem
        from rdkit.Chem import BRICS
        from collections import Counter
        import multiprocessing
        from collections import Counter

        def process_substructures(smiles_chunk):
            substructures = Counter()
            for smiles in smiles_chunk:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Get all possible substructures using BRICS algorithm
                    fragments = BRICS.BRICSDecompose(mol)
                    cleaned_fragments = {clean_substructure(frag) for frag in fragments}
                    substructures.update(cleaned_fragments)
            return substructures

        def clean_substructure(fragment):
            import re
            return re.sub(r'\[\d+\*\]', '', fragment)

        def extract_substructures(smiles_list, num_workers=None):
            if num_workers is None:
                num_workers = multiprocessing.cpu_count()

            chunk_size = len(smiles_list) // num_workers + 1
            print("chunk_size:", chunk_size)
            chunks = [smiles_list[i:i + chunk_size] for i in range(0, len(smiles_list), chunk_size)]

            # Set up a multiprocessing pool and process the chunks
            # Use tqdm to show the progress bar for the chunks being processed
            with multiprocessing.Pool(num_workers) as pool:
                results = []
                for result in tqdm(pool.imap(process_substructures, chunks), total=len(chunks),
                                   desc="Processing SMILES"):
                    results.append(result)

            # Combine results from all workers into a single Counter
            total_substructures = Counter()
            for result in results:
                total_substructures.update(result)

            return total_substructures

        save_path = f"./test_scaffolds_{cfg.dataset_name.split('/')[-1]}.csv"
        result_test = set(pd.read_csv(save_path, index_col=0)['test_scaffolds'])
        save_path = f"./valid_scaffolds_{cfg.dataset_name.split('/')[-1]}.csv"
        result_valid = set(pd.read_csv(save_path, index_col=0)['valid_scaffolds'])
        save_path = f"./train_scaffolds_{cfg.dataset_name.split('/')[-1]}.csv"
        result_train = set(pd.read_csv(save_path, index_col=0)['train_scaffolds'])

        common_elements = result_test.intersection(result_valid)
        print(f"number of common element between scaffolds test and valid set: {len(common_elements)}")
        print(common_elements)
        common_elements = result_train.intersection(result_valid)
        print(f"number of common element between scaffolds train and valid set: {len(common_elements)}")
        print(common_elements)
        common_elements = result_train.intersection(result_test)
        print(f"number of common element between scaffolds train and test set: {len(common_elements)}")
        print(common_elements)

        print('=========================extract substructures test set=========================')
        substructures_testset = extract_substructures(list(result_test))
        save_path = f"./substructures_test_{cfg.dataset_name.split('/')[-1]}.csv"
        substructures_result = pd.DataFrame(list(substructures_testset.items()), columns=['Substructure', 'Count'])
        substructures_result.to_csv(save_path)
        print(f"save substructures result in: {save_path}")

        print('=========================extract substructures valid set=========================')
        substructures_validset = extract_substructures(list(result_valid))
        save_path = f"./substructures_valid_{cfg.dataset_name.split('/')[-1]}.csv"
        substructures_result = pd.DataFrame(list(substructures_validset.items()), columns=['Substructure', 'Count'])
        substructures_result.to_csv(save_path)
        print(f"save substructures result in: {save_path}")

        print('=========================extract substructures train set=========================')
        substructures_trainset = extract_substructures(list(result_train))
        save_path = f"./substructures_train_{cfg.dataset_name.split('/')[-1]}.csv"
        substructures_result = pd.DataFrame(list(substructures_trainset.items()), columns=['Substructure', 'Count'])
        substructures_result.to_csv(save_path)
        print(f"save substructures result in: {save_path}")

        substructures_testset = set(substructures_testset.keys())
        substructures_validset = set(substructures_validset.keys())
        substructures_trainset = set(substructures_trainset.keys())
        common_elements = substructures_testset.intersection(substructures_validset)
        print(f"number of common element between substructures test and valid set: {len(common_elements)}")
        print(common_elements)
        common_elements = substructures_trainset.intersection(substructures_validset)
        print(f"number of common element between substructures train and valid set: {len(common_elements)}")
        print(common_elements)
        common_elements = substructures_trainset.intersection(substructures_testset)
        print(f"number of common element between substructures train and test set: {len(common_elements)}")
        print(common_elements)

    elif mode == 'len-histogram':
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

    else:
        raise NotImplementedError

if __name__ == "__main__":
    datamodule = entrypoint()
