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

DOCKING_SCORE_RESULT_PATH = 'docking_scores.csv'


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
        docking_metrics = pd.read_csv(save_path, index_col=0)
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
        docking_metrics['fa7'] = np.zeros(len(df['SMILES']))
        docking_metrics['parp1'] = np.zeros(len(df['SMILES']))
        docking_metrics['5ht1b'] = np.zeros(len(df['SMILES']))
        docking_metrics['jak2'] = np.zeros(len(df['SMILES']))
        docking_metrics['braf'] = np.zeros(len(df['SMILES']))
        docking_metrics.to_csv(save_path)

    indices = np.where(docking_metrics[args.target] == 1000)[0]
    if indices.size > 0:
        last_index = indices[-1]
    else:
        last_index = 0

    print(f"start compute metric for target {args.target} from {last_index} to {last_index+args.batch_size}")
    docking_metrics[args.target][last_index+args.batch_size+1] = 1000

    docking_cfg = DockingConfig(target_name=args.target, num_sub_proc=args.preprocess_num_jobs,
                                num_cpu_dock=1, seed=args.seed)
    target = DockingVina(docking_cfg)

    st = time.time()
    new_smiles_scores = target.predict(docking_metrics['SMILES'][last_index:last_index + args.batch_size])
    print(f'finish docking in {time.time() - st} seconds')

    docking_metrics.loc[last_index:last_index + args.batch_size, args.target] = new_smiles_scores

    top10_df = pd.DataFrame({col: docking_metrics[col].nsmallest(10).values for col in docking_metrics})
    print(top10_df)
    target.__del__()


