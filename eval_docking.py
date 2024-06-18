import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import os
from hydra import compose, initialize
from utils import creat_unique_experiment_name
from docking_score import DockingConfig, DockingVina
from moses.metrics.metrics import canonic_smiles
from moses.utils import disable_rdkit_log, mapper
import time
import portalocker
import signal
import sys
import re

DOCKING_SCORE_RESULT_PATH = 'docking_scores/docking_scores.csv'


def handle_sigterm(signum, frame):
    print("SIGTERM received. Saving state and exiting...")
    print('FAILED')
    sys.exit(0)


# Register the signal handler
signal.signal(signal.SIGTERM, handle_sigterm)


def args_parser():
    parser = argparse.ArgumentParser(
        description='Binding Affinity Score',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config_name', default="train_ZINC_270M_atomwise", type=str,
                        help='name of the trained config')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--start_index', default=0, type=int, help='start index to compute metrics')
    parser.add_argument('--preprocess_num_jobs', default=8, type=int, help='preprocess_num_jobs')
    parser.add_argument('--target', default='fa7', type=str, help='task to filter for')
    args = parser.parse_args()
    return args


def read_df_safe(file_path):
    with open(file_path, 'r') as file:
        portalocker.lock(file, portalocker.LOCK_SH)  # Shared lock for reading
        df = pd.read_csv(file, index_col=0)
        portalocker.unlock(file)
        return df


def save_df(df, file_path, indices, target_column, new_values):
    # Create a new DataFrame for the specific indices
    update_df = df.loc[indices].copy()
    update_df[target_column] = new_values

    # Construct the new file name
    new_file_path = f"{file_path.split('.csv')[0]}_{target_column}_{indices[-1]}.csv"

    # Save the new DataFrame to the new file
    update_df.to_csv(new_file_path)
    print(f"Saved updated indices {indices} to {new_file_path}")
    return update_df


def read_and_merge_files(output_dir, target):
    # Pattern to match filenames
    pattern = re.compile(rf'docking_scores_{target}_(\d+)\.csv')

    # Function to extract integer from filename
    def extract_number(filename):
        match = re.search(rf'docking_scores_{target}_(\d+)\.csv', filename)
        return int(match.group(1)) if match else float('inf')

    # List to hold DataFrames
    df_list = []

    # Get all files and sort by extracted number
    files = os.listdir(output_dir)
    sorted_files = sorted(files, key=extract_number)

    # Iterate over sorted filenames
    for filename in sorted_files:
        if pattern.match(filename):
            # Construct full file path
            file_path = os.path.join(output_dir, filename)
            # Read the DataFrame
            df = pd.read_csv(file_path, index_col=0)
            # Append to the list of DataFrames
            df_list.append(df)

    # Concatenate all DataFrames
    merged_df = pd.concat(df_list, ignore_index=True)

    merged_df.to_csv(os.path.join(output_dir, f'{target}_final.csv'))

    return merged_df


def save_all_results(output_dir):
    targets = ['fa7', 'parp1', '5ht1b', 'jak2', 'braf']
    merged_table = pd.DataFrame()
    for target in targets:
        df = read_and_merge_files(output_dir, target)
        if merged_table.empty:
            merged_table = df
        else:
            merged_table = pd.merge(merged_table, df, on='SMILES', how='outer')

    merged_table.to_csv(os.path.join(output_dir, f'docking_scores_final.csv'))

def entrypoint(args):
    # Initialize setup
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name=args.config_name)

    exp_name = creat_unique_experiment_name(cfg)
    if args.ckpt is None:
        output_dir = os.path.join(cfg.save_path, exp_name)
    else:
        output_dir = os.path.join(cfg.save_path, exp_name, args.ckpt)

    if not os.path.exists(output_dir):
        raise f"cannot find the experiment in {output_dir}"

    save_path = os.path.join(output_dir, DOCKING_SCORE_RESULT_PATH)
    if os.path.exists(save_path):
        print(f'load generated smiles from {save_path}')
        docking_metrics = read_df_safe(save_path)
    else:
        os.makedirs(os.path.join(output_dir, 'docking_scores'), exist_ok=True)
        print(f"select valid and unique molecules and save in {save_path}")
        df = pd.read_csv(os.path.join(output_dir, 'generated_smiles_42.csv'))
        docking_metrics = pd.DataFrame()
        disable_rdkit_log()
        # remove invalid molecules
        smiles = [x for x in mapper(args.preprocess_num_jobs)(canonic_smiles, list(df['SMILES'])) if x is not None]
        # select unique and valid molecules
        new_smiles = list(set(smiles) - {None})
        docking_metrics['SMILES'] = np.array(new_smiles)
        docking_metrics.to_csv(save_path)

    print(
        f"start compute metric for target {args.target} from {args.start_index} to {args.start_index + args.batch_size}")
    docking_cfg = DockingConfig(target_name=args.target, num_sub_proc=args.preprocess_num_jobs,
                                num_cpu_dock=1, seed=args.seed)
    target = DockingVina(docking_cfg)

    st = time.time()

    try:
        new_smiles_scores = target.predict(
            docking_metrics['SMILES'][args.start_index:args.start_index + args.batch_size])
    except Exception as e:
        print(f'FAILED: {str(e)}')
        sys.exit(1)

    print(f'finish docking in {time.time() - st} seconds')
    update_df = save_df(docking_metrics, save_path, range(args.start_index, args.start_index + args.batch_size),
                        args.target, new_smiles_scores)
    negative_values = update_df[args.target][update_df[args.target] < 0]
    res_ = negative_values.nsmallest(int(0.05 * len(negative_values))).mean()
    print(f'Average top 5% of {args.target}: {res_}')

    target.__del__()


if __name__ == "__main__":
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    args = args_parser()
    entrypoint(args)
