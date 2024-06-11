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

DOCKING_SCORE_RESULT_PATH = 'docking_scores.csv'
# Global variables to be used in signal handler
docking_metrics = None
last_index = None
args = None
save_path = None

def handle_sigterm(signum, frame):
    print("SIGTERM received. Saving state and exiting...")
    if docking_metrics is not None and last_index is not None and args is not None and save_path is not None:
        docking_metrics.loc[last_index + args.batch_size, args.target] = 0
        save_df(docking_metrics, save_path)
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
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
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

def save_df_safe(df, file_path):
    with open(file_path, 'r+') as file:
        portalocker.lock(file, portalocker.LOCK_EX)  # Exclusive lock for writing
        file.seek(0)
        df.to_csv(file)
        file.truncate()  # Important to truncate in case new file is shorter
        portalocker.unlock(file)

def save_df(df, file_path, indices, target_column, new_values):
    # Create a new DataFrame for the specific indices
    update_df = df.loc[indices, [target_column]].copy()
    update_df[target_column] = new_values

    # Construct the new file name
    new_file_path = f"{file_path}_{target_column}_{indices[-1]}"

    # Save the new DataFrame to the new file
    update_df.to_csv(new_file_path)
    print(f"Saved updated indices {indices} to {new_file_path}")


def average_of_lowest_negatives(column):
    # Filter the column to keep only negative values
    negative_values = column[column < 0]
    # Get the top 10 lowest negative values and compute their average
    return negative_values.nsmallest(10).mean()


def entrypoint(args):
    # Initialize setup
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name=args.config_name)

    exp_name = creat_unique_experiment_name(cfg)
    output_dir = os.path.join(cfg.save_path, exp_name)
    if not os.path.exists(output_dir):
        raise f"cannot find the experiment in {output_dir}"

    save_path = os.path.join(output_dir, DOCKING_SCORE_RESULT_PATH)
    if os.path.exists(save_path):
        print(f'load generated smiles from {save_path}')
        docking_metrics = read_df_safe(save_path)
    else:
        print(f"select valid and unique molecules and save in {save_path}")
        df = pd.read_csv(os.path.join(output_dir, 'generated_smiles_42.csv'), index_col=0)
        docking_metrics = pd.DataFrame()
        disable_rdkit_log()
        # remove invalid molecules
        smiles = [x for x in mapper(args.preprocess_num_jobs)(canonic_smiles, list(df['SMILES'])) if x is not None]
        # select unique and valid molecules
        new_smiles = list(set(smiles) - {None})
        docking_metrics['SMILES'] = np.array(new_smiles)
        docking_metrics['fa7'] = np.zeros(len(new_smiles))
        docking_metrics['parp1'] = np.zeros(len(new_smiles))
        docking_metrics['5ht1b'] = np.zeros(len(new_smiles))
        docking_metrics['jak2'] = np.zeros(len(new_smiles))
        docking_metrics['braf'] = np.zeros(len(new_smiles))
        docking_metrics.to_csv(save_path)

    indices = np.where(docking_metrics[args.target] == 1000)[0]
    if indices.size > 0:
        last_index = indices[-1]
    else:
        last_index = 0

    print(f"start compute metric for target {args.target} from {last_index} to {last_index + args.batch_size}")
    docking_metrics.loc[last_index + args.batch_size, args.target] = 1000
    save_df_safe(docking_metrics, save_path)

    docking_cfg = DockingConfig(target_name=args.target, num_sub_proc=args.preprocess_num_jobs,
                                num_cpu_dock=1, seed=args.seed)
    target = DockingVina(docking_cfg)

    try:
        st = time.time()
        new_smiles_scores = target.predict(docking_metrics['SMILES'][last_index:last_index + args.batch_size])
        print(f'finish docking in {time.time() - st} seconds')
        save_df(docking_metrics, save_path, range(last_index, last_index + args.batch_size), args.target, new_smiles_scores)
    except:
        docking_metrics.loc[last_index + args.batch_size, args.target] = 0
        save_df_safe(docking_metrics, save_path)
        print('FAILED')

    negative_values = docking_metrics[args.target][docking_metrics[args.target] < 0]
    res_ = negative_values.nsmallest(int(0.05 * len(docking_metrics['SMILES']))).mean()
    print(f'Average top 5% of {args.target}: {res_}')
    target.__del__()


if __name__ == "__main__":
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    args = args_parser()
    entrypoint(args)
