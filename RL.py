from dataset import MolGenDataModule
from models import GPT2MolGen, GPT2MolGen_flash_atten, Llama_small_flash_atten
from utils import creat_unique_experiment_name
from transformers import GPT2LMHeadModel, LlamaForCausalLM, LlamaConfig, set_seed
from docking_score import DockingConfig, DockingVina
from accelerate import Accelerator
from hydra import compose, initialize
import argparse
import numpy as np
import wandb
import torch
import os
import sys

import tdc
import yaml
from tdc import Oracle
from rdkit import Chem
import time

import torch.nn.functional as F
import torch.distributions as td



def top_auc(buffer, top_n, finish, env_log_interval, max_oracle_calls):
    sum = 0
    prev = 0
    called = 0
    ordered_results = list(
        sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False)
    )
    for idx in range(
            env_log_interval, min(len(buffer), max_oracle_calls), env_log_interval
    ):
        temp_result = ordered_results[:idx]
        temp_result = list(sorted(temp_result, key=lambda kv: kv[1][0], reverse=True))[
                      :top_n
                      ]
        top_n_now = np.mean([item[1][0] for item in temp_result])
        sum += env_log_interval * (top_n_now + prev) / 2
        prev = top_n_now
        called = idx
    temp_result = list(sorted(ordered_results, key=lambda kv: kv[1][0], reverse=True))[
                  :top_n
                  ]
    top_n_now = np.mean([item[1][0] for item in temp_result])
    sum += (len(buffer) - called) * (top_n_now + prev) / 2
    if finish and len(buffer) < max_oracle_calls:
        sum += (max_oracle_calls - len(buffer)) * top_n_now
    return sum / max_oracle_calls


class BaseOptimizer:
    def __init__(self, target_name='fa7', max_oracle_calls=10000, env_log_interval=100, wandb_log=False):
        self.target_name = target_name
        # defining target oracles
        self.assign_target(task='docking')
        print("Target is assigned")

        # defining standard oracles
        self.sa_scorer = tdc.Oracle(name="SA")
        self.diversity_evaluator = tdc.Evaluator(name="Diversity")
        self.filter = tdc.chem_utils.oracle.filter.MolFilter(
            filters=["PAINS", "SureChEMBL", "Glaxo"], property_filters_flag=False
        )

        self.max_oracle_calls = max_oracle_calls
        self.env_log_interval = env_log_interval

        # store all unique molecules
        self.mol_buffer = dict()

        self.mean_score = 0

        # logging counters
        self.last_log = 0
        self.last_log_time = time.time()
        self.total_count = 0
        self.invalid_count = 0
        self.redundant_count = 0
        self.wandb_log = wandb_log
        self.task = 'docking'
        self.output_dir = '.'

        print("Initialisation of base optimizer is done!")

    @property
    def budget(self):
        return self.max_oracle_calls

    @property
    def finish(self):
        return len(self.mol_buffer) >= self.max_oracle_calls

    def assign_target(self, task='docking'):
        if task == "docking":

            docking_cfg = DockingConfig(target_name=self.target_name, num_sub_proc=24,
                                        num_cpu_dock=1, seed=42)
            self.target = DockingVina(docking_cfg)
            self.predict = self.predict_docking

        elif task == "augmented_docking":

            docking_config = dict()
            if self.target_name == "fa7":
                box_center = (10.131, 41.879, 32.097)
                box_size = (20.673, 20.198, 21.362)
            elif self.target_name == "parp1":
                box_center = (26.413, 11.282, 27.238)
                box_size = (18.521, 17.479, 19.995)
            elif self.target_name == "5ht1b":
                box_center = (-26.602, 5.277, 17.898)
                box_size = (22.5, 22.5, 22.5)
            elif self.target_name == "jak2":
                box_center = (114.758, 65.496, 11.345)
                box_size = (19.033, 17.929, 20.283)
            elif self.target_name == "braf":
                box_center = (84.194, 6.949, -7.081)
                box_size = (22.032, 19.211, 14.106)
            else:
                raise NotImplementedError

            docking_config["receptor_file"] = f'./docking_score/receptors/{self.target_name}/receptor.pdbqt'
            box_parameter = (box_center, box_size)
            docking_config["box_parameter"] = box_parameter
            docking_config["vina_program"] = 'qvina2'
            docking_config["temp_dir"] = f'tmp/tmp{2}'
            docking_config["exhaustiveness"] = 1
            docking_config["num_sub_proc"] = 8
            docking_config["num_cpu_dock"] = 1
            docking_config["num_modes"] = 10
            docking_config["timeout_gen3d"] = 30
            docking_config["timeout_dock"] = 100

            self.target = DockingVina(docking_config)
            self.qed_scorer = Oracle(name="qed")
            self.predict = self.predict_augmented_docking

        elif task == "pmo":
            self.target = Oracle(name=self.target_name)
            self.predict = self.predict_pmo
        else:
            raise NotImplementedError

    def define_wandb_metrics(self):
        # new wandb metric
        wandb.define_metric("num_molecules")
        wandb.define_metric("avg_top1", step_metric="num_molecules")
        wandb.define_metric("avg_top10", step_metric="num_molecules")
        wandb.define_metric("avg_top100", step_metric="num_molecules")
        wandb.define_metric("auc_top1", step_metric="num_molecules")
        wandb.define_metric("auc_top10", step_metric="num_molecules")
        wandb.define_metric("auc_top100", step_metric="num_molecules")
        wandb.define_metric("avg_sa", step_metric="num_molecules")
        wandb.define_metric("diversity_top100", step_metric="num_molecules")
        wandb.define_metric("n_oracle", step_metric="num_molecules")
        wandb.define_metric("invalid_count", step_metric="num_molecules")
        wandb.define_metric("redundant_count", step_metric="num_molecules")

    def score_pmo(self, smi):
        """
        Function to score one molecule
        Argguments:
            smi: One SMILES string represnets a moelcule.
        Return:
            score: a float represents the property of the molecule.
        """
        if len(self.mol_buffer) > self.max_oracle_calls:
            return 0
        if smi is None:
            return 0
        mol = Chem.MolFromSmiles(smi)
        if mol is None or len(smi) == 0:
            self.invalid_count += 1
            return 0.0
        else:
            smi = Chem.MolToSmiles(mol)
            if smi in self.mol_buffer:
                self.mol_buffer[smi][2] += 1
                self.redundant_count += 1
            else:
                self.mol_buffer[smi] = [
                    float(self.target(smi)),
                    len(self.mol_buffer) + 1,
                    1,
                ]
            return self.mol_buffer[smi][0]

    def predict_pmo(self, smiles_list):
        st = time.time()
        assert isinstance(smiles_list, list)
        self.total_count += len(smiles_list)
        score_list = []
        for smi in smiles_list:
            score_list.append(self.score_pmo(smi))
            if (
                    len(self.mol_buffer) % self.env_log_interval == 0
                    and len(self.mol_buffer) > self.last_log
            ):
                self.sort_buffer()
                self.log_intermediate()
                self.last_log_time = time.time()
                self.last_log = len(self.mol_buffer)

        self.last_logging_time = time.time() - st
        self.mean_score = np.mean(score_list)
        return score_list

    def predict_augmented_docking(self, smiles_list):
        """
        Score
        """
        st = time.time()
        assert isinstance(smiles_list, list)
        self.total_count += len(smiles_list)
        score_list = [None] * len(smiles_list)
        new_smiles = []
        new_smiles_ptrs = []
        for i, smi in enumerate(smiles_list):
            if smi in self.mol_buffer:
                score_list[i] = self.mol_buffer[smi][0]
                self.mol_buffer[smi][2] += 1
                self.redundant_count += 1
            else:
                new_smiles.append((smi))
                new_smiles_ptrs.append((i))

        new_smiles_scores = self.target(new_smiles)

        for smi, ptr, sc in zip(new_smiles, new_smiles_ptrs, new_smiles_scores):
            if sc == 99.0:
                self.invalid_count += 1
                sc = 0
            self.mol_buffer[smi] = [
                (-sc / 20) * ((10 - self.sa_scorer(smi)) / 9) * self.qed_scorer(smi),
                len(self.mol_buffer) + 1,
                1,
                -sc,
            ]
            score_list[ptr] = self.mol_buffer[smi][0]

            if (
                    len(self.mol_buffer) % self.env_log_interval == 0
                    and len(self.mol_buffer) > self.last_log
            ):
                self.sort_buffer()
                self.log_intermediate()
                self.last_log_time = time.time()
                self.last_log = len(self.mol_buffer)

        self.last_logging_time = time.time() - st
        self.mean_score = np.mean(score_list)
        return score_list

    def predict_docking(self, smiles_list):
        """
        Score
        """
        st = time.time()
        assert isinstance(smiles_list, list)
        self.total_count += len(smiles_list)
        score_list = [None] * len(smiles_list)
        new_smiles = []
        new_smiles_ptrs = []
        for i, smi in enumerate(smiles_list):
            if smi in self.mol_buffer:
                score_list[i] = self.mol_buffer[smi][0] / 20
                self.mol_buffer[smi][2] += 1
                self.redundant_count += 1
            else:
                new_smiles.append((smi))
                new_smiles_ptrs.append((i))

        new_smiles_scores = self.target.predict(new_smiles)

        for smi, ptr, sc in zip(new_smiles, new_smiles_ptrs, new_smiles_scores):
            if sc == 99.0:
                self.invalid_count += 1
                sc = 0

            self.mol_buffer[smi] = [-sc, len(self.mol_buffer) + 1, 1]
            score_list[ptr] = -sc / 20

            if (
                    len(self.mol_buffer) % self.env_log_interval == 0
                    and len(self.mol_buffer) > self.last_log
            ):
                self.sort_buffer()
                self.log_intermediate()

                self.last_log_time = time.time()
                self.last_log = len(self.mol_buffer)

        self.last_logging_time = time.time() - st
        self.mean_score = np.mean(score_list)
        return score_list

    def optimize(self, cfg):
        raise NotImplementedError

    def sanitize(self, mol_list):
        new_mol_list = []
        smiles_set = set()
        for mol in mol_list:
            if mol is not None:
                try:
                    smiles = Chem.MolToSmiles(mol)
                    if smiles is not None and smiles not in smiles_set:
                        smiles_set.add(smiles)
                        new_mol_list.append(mol)
                except ValueError:
                    print("bad smiles")
        return new_mol_list

    def sort_buffer(self):
        self.mol_buffer = dict(
            sorted(self.mol_buffer.items(), key=lambda kv: kv[1][0], reverse=True)
        )

    def log_intermediate(self, mols=None, scores=None, finish=False):
        if finish:
            temp_top100 = list(self.mol_buffer.items())[:100]
            smis = [item[0] for item in temp_top100]
            scores = [item[1][0] for item in temp_top100]
            if self.task == "augmented_docking":
                docking_scores = [item[1][3] for item in temp_top100]
            n_calls = self.max_oracle_calls

        else:
            if mols is None and scores is None:
                if len(self.mol_buffer) <= self.max_oracle_calls:
                    # If not spefcified, log current top-100 mols in buffer
                    temp_top100 = list(self.mol_buffer.items())[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    if self.task == "augmented_docking":
                        docking_scores = [item[1][3] for item in temp_top100]
                    else:
                        docking_scores = [0] * len(scores)
                    n_calls = len(self.mol_buffer)
                else:
                    results = list(
                        sorted(
                            self.mol_buffer.items(),
                            key=lambda kv: kv[1][1],
                            reverse=False,
                        )
                    )[: self.max_oracle_calls]
                    temp_top100 = sorted(
                        results, key=lambda kv: kv[1][0], reverse=True
                    )[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    if self.task == "augmented_docking":
                        docking_scores = [item[1][3] for item in temp_top100]
                    else:
                        docking_scores = [0] * len(scores)
                    n_calls = self.max_oracle_calls
            else:
                # Otherwise, log the input moleucles
                smis = [Chem.MolToSmiles(m) for m in mols]
                n_calls = len(self.mol_buffer)

        avg_top1 = np.max(scores)
        avg_top10 = np.mean(sorted(scores, reverse=True)[:10])
        avg_top100 = np.mean(scores)

        avg_docking_top1 = np.max(docking_scores)
        avg_docking_top10 = np.mean(sorted(docking_scores, reverse=True)[:10])
        avg_docking_top100 = np.mean(docking_scores)

        avg_sa = np.mean(self.sa_scorer(smis))
        diversity_top100 = self.diversity_evaluator(smis)
        print(
            f"{n_calls}/{self.max_oracle_calls} | "
            f"avg_top1: {avg_top1:.3f} | "
            f'avg_top10: {avg_top10:.3f} | '
            f'avg_top100: {avg_top100:.3f} | '
            f"time: {time.time() - self.last_log_time:.3f} | "
            f"mean_score: {self.mean_score:.3f} | "
            f"tot_cnt: {self.total_count} | "
            f"inv_count: {self.invalid_count} | "
            f"red_cnt: {self.redundant_count} | "
        )

        if self.wandb_log:
            wandb.log(
                {
                    "avg_top1": avg_top1,
                    "avg_top10": avg_top10,
                    "avg_top100": avg_top100,
                    "avg_docking_top1": avg_docking_top1,
                    "avg_docking_top10": avg_docking_top10,
                    "avg_docking_top100": avg_docking_top100,
                    "auc_top1": top_auc(
                        self.mol_buffer,
                        1,
                        finish,
                        self.env_log_interval,
                        self.max_oracle_calls,
                    ),
                    "auc_top10": top_auc(
                        self.mol_buffer,
                        10,
                        finish,
                        self.env_log_interval,
                        self.max_oracle_calls,
                    ),
                    "auc_top100": top_auc(
                        self.mol_buffer,
                        100,
                        finish,
                        self.env_log_interval,
                        self.max_oracle_calls,
                    ),
                    "avg_sa": avg_sa,
                    "diversity_top100": diversity_top100,
                    "invalid_count": self.invalid_count,
                    "redundant_count": self.redundant_count,
                    "num_molecules": n_calls,
                }
            )

            # data = [[scores[i], docking_scores[i], smis[i], wandb.Image(Draw.MolToImage(Chem.MolFromSmiles(smis[i])))] for i in range(10)]

            # columns = ["Score", "Docking score", "SMILES", "IMAGE"]
            # wandb.log({"Top 10 Molecules": wandb.Table(data=data, columns=columns)})

    def _analyze_results(self, results):
        results = results[:100]
        scores_dict = {item[0]: item[1][0] for item in results}
        smis = [item[0] for item in results]
        scores = [item[1][0] for item in results]
        smis_pass = self.filter(smis)
        if len(smis_pass) == 0:
            top1_pass = -1
        else:
            top1_pass = np.max([scores_dict[s] for s in smis_pass])
        return [
            np.mean(scores),
            np.mean(scores[:10]),
            np.max(scores),
            self.diversity_evaluator(smis),
            np.mean(self.sa_scorer(smis)),
            float(len(smis_pass) / 100),
            top1_pass,
        ]

    def save_result(self, suffix=None):
        if suffix is None:
            output_file_path = os.path.join(self.output_dir, "results.yaml")
        else:
            output_file_path = os.path.join(
                self.output_dir, "results_" + suffix + ".yaml"
            )

        self.sort_buffer()
        with open(output_file_path, "w") as f:
            yaml.dump(self.mol_buffer, f, sort_keys=False)

    def __len__(self):
        return len(self.mol_buffer)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class LlamaPolicy(object):
    def __init__(self, model, tokenizer, max_len: int, device: str):
        super(LlamaPolicy, self).__init__()
        self.max_len = max_len
        self.device = device
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, x):
        logits = self.model(x).logits
        return td.Categorical(logits=logits)

    def autoregress(self, x):
        logits = self.model(x).logits[-1]
        return td.Categorical(logits=logits)

    def sample(self, batch_size, max_length):
        assert max_length <= self.max_len
        preds = self.tokenizer.bos_token_id * torch.ones((1, batch_size), dtype=torch.long, device=self.device)
        finished = torch.zeros((batch_size), dtype=torch.bool, device=self.device)
        imag_smiles_lens = torch.ones((batch_size), device=self.device)

        with torch.no_grad():
            for i in range(1, max_length + 1):
                preds_dist = self.forward(preds)
                next_preds = preds_dist.sample()[-1].view(1, -1)
                preds = torch.cat([preds, next_preds], dim=0)
                imag_smiles_lens += ~finished

                EOS_sampled = (preds[-1] == self.tokenizer.eos_token_id)
                finished = torch.ge(finished + EOS_sampled, 1)
                if torch.prod(finished) == 1:
                    break

        imag_smiles = preds.T.tolist()
        return imag_smiles, imag_smiles_lens[0].tolist()

    def get_likelihood(self, obs, nonterms):
        dist = self.forward(obs[:-1])
        logprobs = dist.log_prob(obs[1:]) * nonterms[:-1]
        # print(logprobs.shape)
        # print(dist.logits.shape)
        # print(dist.probs.shape)
        # exit()
        log_of_probs = F.log_softmax(dist.logits, dim=-1) * nonterms[:-1].unsqueeze(-1)
        action_probs = dist.probs * nonterms[:-1].unsqueeze(-1)

        # print(log_of_probs)
        # print(action_probs)
        # print(torch.log(action_probs))
        # exit()

        return logprobs, log_of_probs, action_probs

    def get_data(self, batch_size, max_length):
        if max_length is None:
            max_length = self.max_len
        else:
            assert max_length <= self.max_len

        input_ids = self.tokenizer.encode("<bos>", return_tensors="pt").to(self.device)  # type: ignore
        input_ids = input_ids.repeat_interleave(batch_size, dim=0)
        # skip eos token
        input_ids = input_ids[:, :-2]
        attention_mask = torch.ones_like(input_ids).to(self.device)
        output = self.model.generate(
            input_ids,
            max_length=max_length,
            top_k=50,
            top_p=0.95,
            temperature=1,
            eos_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )
        obs = output.sequences.T
        nonterms = torch.zeros((max_length + 1, batch_size), dtype=torch.bool, device=self.device)
        rewards = torch.zeros((max_length, batch_size), dtype=torch.float32, device=self.device)
        end_flags = torch.zeros((1, batch_size), dtype=torch.bool, device=self.device)

        for i in range(1, max_length + 1):
            preds = obs[i]
            nonterms[i - 1] = ~end_flags

            EOS_sampled = (preds == self.tokenizer.eos_token_id)
            rewards[i - 1] = EOS_sampled * (~end_flags)

            # check if all sequences are done
            end_flags = torch.ge(end_flags + EOS_sampled, 1)
            if torch.prod(end_flags) == 1:
                break

        if i == max_length:
            rewards[-1] = rewards[-1] + (~end_flags)

        # remove assertion afterwards
        assert rewards.sum() == batch_size

        obs = obs[:i + 1]
        nonterms = nonterms[:i + 1]
        rewards = rewards[:i]
        episode_lens = nonterms.sum(0).cpu()

        return obs.clone(), rewards.clone(), nonterms.clone(), episode_lens


def get_params(model):
    return (p for p in model.parameters() if p.requires_grad)


def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()


class reinforce_optimizer(BaseOptimizer):
    def __init__(self, config_name='train_ZINC_270M_atomwise', ckpt: int = None, **kwargs):
        super().__init__(**kwargs)
        self.config_name = config_name
        self.ckpt = ckpt

    def _init(self):

        # Initialize setup
        with initialize(version_base=None, config_path="configs"):
            cfg = compose(config_name=self.config_name)

        exp_name = creat_unique_experiment_name(cfg)
        output_dir = os.path.join(cfg.save_path, exp_name)
        self.output_dir = output_dir

        # Initialize DataModule
        cfg.dataset.dataset_name = "MolGen/ZINC_270M-raw"
        datamodule = MolGenDataModule(**cfg.dataset)

        # Initialize model
        if cfg.model.model_name_or_path == 'gpt2_flash_atten':
            model = GPT2MolGen_flash_atten(**cfg.model, max_seq_length=datamodule.max_seq_length,
                                           vocab_size=datamodule.tokenizer.vocab_size)
        elif cfg.model.model_name_or_path in ['llama_small', 'llama_small_HF']:
            model_cfg = LlamaConfig(**cfg.model, n_ctx=datamodule.max_seq_length,
                                    vocab_size=datamodule.tokenizer.vocab_size)
            model = LlamaForCausalLM(model_cfg)
        elif cfg.model.model_name_or_path == 'llama_small_FA':
            model = Llama_small_flash_atten(**cfg.model, max_seq_length=datamodule.max_seq_length,
                                            vocab_size=datamodule.tokenizer.vocab_size)
        elif cfg.model.model_name_or_path == 'gpt2':
            model = GPT2MolGen(**cfg.model, max_seq_length=datamodule.max_seq_length,
                               vocab_size=datamodule.tokenizer.vocab_size)
        else:
            raise NotImplementedError

        if self.ckpt is None:
            checkpoint_path = os.path.join(output_dir, 'pytorch_model.bin')
        else:
            checkpoint_path = os.path.join(output_dir, f'tmp-spec-checkpoint-{self.ckpt}', 'pytorch_model.bin')
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        print(f'***************** load checkpoint from {checkpoint_path} *****************')

        #################################################################

        # get agent
        self.agent = LlamaPolicy(model, datamodule.tokenizer, max_len=64, device='cuda')
        print("Agent class initialised")

        self.vocab = self.agent.tokenizer.get_vocab()
        print("Vocab assigned")

        self.target_entropy = -0.98 * torch.log(1 / torch.tensor(len(self.vocab)))
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp().item()
        self.a_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4, eps=1e-4)

        # get optimizers
        self.optimizer = torch.optim.Adam(
            get_params(self.agent.model), lr=1e-4
        )
        accelerator = Accelerator(mixed_precision='bf16', device_placement='cuda')
        self.agent.model, self.optimizer, self.a_optimizer = accelerator.prepare(self.agent.model,
                                                                                 self.optimizer,
                                                                                 self.a_optimizer)
        print("Initialisation of optimizer is done!")

    def update(self, obs, rewards, nonterms, episode_lens, metrics, log):
        rev_returns = torch.cumsum(rewards, dim=0)
        advantages = rewards - rev_returns + rev_returns[-1:]

        logprobs, log_of_probs, action_probs = self.agent.get_likelihood(obs, nonterms)

        loss_pg = -advantages * logprobs
        loss_pg = loss_pg.sum(0, keepdim=True).mean()

        # loss_p = - (1 / logprobs.sum(0, keepdim=True)).mean()
        loss = loss_pg  # + cfg.lp_coef * loss_p
        loss = loss_pg + self.alpha * logprobs.sum(0, keepdim=True).mean()

        # Calculate gradients and make an update to the network weights
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), 0.5)
        self.optimizer.step()

        alpha_loss = (
                action_probs.detach()
                * (-self.log_alpha.exp() * (log_of_probs + self.target_entropy).detach())
        ).mean()

        self.a_optimizer.zero_grad()
        alpha_loss.backward()
        self.a_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

        if log:
            metrics["pg_loss"] = loss_pg.item()
            metrics["agent_likelihood"] = logprobs.sum(0).mean().item()
            metrics["grad_norm"] = grad_norm.item()
            metrics["smiles_len"] = episode_lens.float().mean().item()
            # metrics['loss_p'] = loss_p.item()
            metrics["alpha"] = self.alpha
            metrics["alpha_loss"] = alpha_loss.detach().item()
            print("logging!")
            wandb.log(metrics)

    def optimize(self, max_strings, batch_size=96, max_len=64):

        # set device
        self.device = torch.device('cuda')
        self._init()

        train_steps = 0
        eval_strings = 0
        metrics = dict()
        print("Start training ... ")
        while eval_strings < max_strings:
            with torch.no_grad():
                # sample experience
                obs, rewards, nonterms, episode_lens = self.agent.get_data(
                    batch_size, max_len
                )

            smiles_list = []
            output = self.agent.tokenizer.batch_decode(obs.T, skip_special_tokens=True)
            smiles_list += [s.replace(" ", "") for s in output]

            score = np.array(self.predict(smiles_list))
            scores = torch.tensor(
                score, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            if self.finish:
                print("max oracle hit")
                wandb.finish()
                sys.exit(0)

            train_steps += 1
            eval_strings += batch_size

            metrics["eval_strings"] = eval_strings
            metrics["mean_score"] = np.mean(scores.cpu().numpy())
            metrics["max_score"] = np.max(scores.cpu().numpy())
            metrics["min_score"] = np.min(scores.cpu().numpy())
            metrics["mean_episode_lens"] = np.mean(episode_lens.tolist())
            metrics["max_episode_lens"] = np.max(episode_lens.tolist())
            metrics["min_episode_lens"] = np.min(episode_lens.tolist())
            print(metrics)

            rewards = rewards * scores
            self.update(obs, rewards, nonterms, episode_lens, metrics, log=self.wandb_log)

            if train_steps % 50 == 0:
                output_dir = os.path.join(self.output_dir, f"reinforce_{self.target_name}")
                os.makedirs(output_dir, exist_ok=True)

                checkpoint = {
                    'eval_strings': eval_strings,
                    'model_state_dict': self.agent.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'a_optimizer_dict': self.a_optimizer.state_dict(),
                    'log_alpha': self.log_alpha,
                }

                check_path = os.path.join(output_dir, f"pytorch_model_{eval_strings}.tar")
                print(f'=============== save checkpoint in {check_path} ===============')
                torch.save(checkpoint, check_path)

        print("max training string hit")
        # wandb.finish()
        sys.exit(0)


def args_parser():
    parser = argparse.ArgumentParser(
        description='Binding Affinity Score',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config_name', default="train_ZINC_270M_atomwise", type=str,
                        help='name of the trained config')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', default=96, type=int, help='batch size')
    parser.add_argument('--max_strings', default=40000, type=int, help='batch size')
    parser.add_argument('--target', default='fa7', type=str, help='task to filter for')
    parser.add_argument('--wandb', action='store_true', help='use wandb for logging')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = args_parser()
    set_seed(args.seed)

    if args.wandb:
        with initialize(version_base=None, config_path="configs"):
            cfg = compose(config_name=args.config_name)
        exp_name = creat_unique_experiment_name(cfg)
        wandb.init(entity='drug-discovery', project='small-molecule-generation',
                   name=f'reinforce_{exp_name}_{args.target}',
                   config={'lr': 1e-4})

    optim = reinforce_optimizer(config_name=args.config_name, ckpt=args.ckpt,
                                target_name=args.target, max_oracle_calls=10000,
                                env_log_interval=96, wandb_log=args.wandb)
    optim.optimize(max_strings=args.max_strings, batch_size=args.batch_size, max_len=64)
