# code modified from https://github.com/SeulLee05/MOOD

from shutil import rmtree
from multiprocessing import Manager
from multiprocessing import Process
from multiprocessing import Queue
import subprocess
from openbabel import pybel
import os
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DockingConfig:
    target_name: str
    vina_program: str = 'qvina2'
    temp_dir: str = field(default_factory=lambda: DockingConfig.make_docking_dir())
    exhaustiveness: int = 1
    num_sub_proc: int = 8
    num_cpu_dock: int = 1
    num_modes: int = 10
    timeout_gen3d: int = 30
    timeout_dock: int = 100
    seed: int = 42
    receptor_file: str = field(init=False)
    box_parameter: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = field(init=False)

    def __post_init__(self):
        self.receptor_file = f'./docking_score/receptors/{self.target_name}/receptor.pdbqt'
        self.box_parameter = self.get_box_parameters()

    def get_box_parameters(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        box_parameters = {
            'fa7': ((10.131, 41.879, 32.097), (20.673, 20.198, 21.362)),
            'parp1': ((26.413, 11.282, 27.238), (18.521, 17.479, 19.995)),
            '5ht1b': ((-26.602, 5.277, 17.898), (22.5, 22.5, 22.5)),
            'jak2': ((114.758, 65.496, 11.345), (19.033, 17.929, 20.283)),
            'braf': ((84.194, 6.949, -7.081), (22.032, 19.211, 14.106))
        }
        return box_parameters.get(self.target_name, (None, None))

    @staticmethod
    def make_docking_dir():
        for i in range(100):
            tmp_dir = f'tmp/tmp{i}'
            if not os.path.exists(tmp_dir):
                print(f'Docking tmp dir: {tmp_dir}')
                os.makedirs(tmp_dir)
                return tmp_dir
        raise ValueError('tmp/tmp0~99 are full. Please delete tmp dirs.')


class DockingVina(object):
    def __init__(self, docking_params):
        super(DockingVina, self).__init__()
        self.temp_dir = docking_params.temp_dir
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

        self.vina_program = docking_params.vina_program
        self.receptor_file = docking_params.receptor_file
        box_parameter = docking_params.box_parameter
        (self.box_center, self.box_size) = box_parameter
        self.exhaustiveness = docking_params.exhaustiveness
        self.num_sub_proc = docking_params.num_sub_proc
        self.num_cpu_dock = docking_params.num_cpu_dock
        self.num_modes = docking_params.num_modes
        self.timeout_gen3d = docking_params.timeout_gen3d
        self.timeout_dock = docking_params.timeout_dock
        self.seed = docking_params.seed

    def gen_3d(self, smi, ligand_mol_file):
        run_line = 'obabel -:%s --gen3D -O %s' % (smi, ligand_mol_file)
        result = subprocess.check_output(run_line.split(),
                                         stderr=subprocess.STDOUT,
                                         timeout=self.timeout_gen3d, universal_newlines=True)

    def docking(self, receptor_file, ligand_mol_file, ligand_pdbqt_file, docking_pdbqt_file):
        ms = list(pybel.readfile("mol", ligand_mol_file))
        m = ms[0]
        m.write("pdbqt", ligand_pdbqt_file, overwrite=True)
        run_line = '%s --receptor %s --ligand %s --out %s' % (self.vina_program,
                                                              receptor_file, ligand_pdbqt_file, docking_pdbqt_file)
        run_line += ' --center_x %s --center_y %s --center_z %s' % (self.box_center)
        run_line += ' --size_x %s --size_y %s --size_z %s' % (self.box_size)
        run_line += ' --cpu %d' % (self.num_cpu_dock)
        run_line += ' --num_modes %d' % (self.num_modes)
        run_line += ' --exhaustiveness %d ' % (self.exhaustiveness)
        run_line += ' --seed %d ' % (self.seed)
        result = subprocess.check_output(run_line.split(),
                                         stderr=subprocess.STDOUT,
                                         timeout=self.timeout_dock, universal_newlines=True)
        result_lines = result.split('\n')

        check_result = False
        affinity_list = list()
        for result_line in result_lines:
            if result_line.startswith('-----+'):
                check_result = True
                continue
            if not check_result:
                continue
            if result_line.startswith('Writing output'):
                break
            if result_line.startswith('Refine time'):
                break
            lis = result_line.strip().split()
            if not lis[0].isdigit():
                break
            affinity = float(lis[1])
            affinity_list += [affinity]
        return affinity_list

    def creator(self, q, data, num_sub_proc):
        for d in data:
            idx = d[0]
            dd = d[1]
            q.put((idx, dd))

        for i in range(0, num_sub_proc):
            q.put('DONE')

    def docking_subprocess(self, q, return_dict, sub_id=0):
        while True:
            qqq = q.get()
            if qqq == 'DONE':
                break
            (idx, smi) = qqq
            # print(smi)
            receptor_file = self.receptor_file
            ligand_mol_file = '%s/ligand_%s.mol' % (self.temp_dir, sub_id)
            ligand_pdbqt_file = '%s/ligand_%s.pdbqt' % (self.temp_dir, sub_id)
            docking_pdbqt_file = '%s/dock_%s.pdbqt' % (self.temp_dir, sub_id)
            try:
                self.gen_3d(smi, ligand_mol_file)
            except Exception as e:
                return_dict[idx] = 99.9
                continue
            try:
                affinity_list = self.docking(receptor_file, ligand_mol_file,
                                             ligand_pdbqt_file, docking_pdbqt_file)
            except Exception as e:
                return_dict[idx] = 99.9
                continue
            if len(affinity_list) == 0:
                affinity_list.append(99.9)

            affinity = affinity_list[0]
            return_dict[idx] = affinity

    def predict(self, smiles_list):
        data = list(enumerate(smiles_list))
        q1 = Queue()
        manager = Manager()
        return_dict = manager.dict()
        proc_master = Process(target=self.creator, args=(q1, data, self.num_sub_proc))
        proc_master.start()

        procs = []
        for sub_id in range(0, self.num_sub_proc):
            proc = Process(target=self.docking_subprocess, args=(q1, return_dict, sub_id))
            procs.append(proc)
            proc.start()

        q1.close()
        q1.join_thread()
        proc_master.join()
        for proc in procs:
            proc.join()
        keys = sorted(return_dict.keys())
        affinity_list = list()
        for key in keys:
            affinity = return_dict[key]
            affinity_list += [affinity]
        return affinity_list

    def __del__(self):
        if os.path.exists(self.temp_dir):
            rmtree(self.temp_dir)
            print(f'{self.temp_dir} removed.')


if __name__ == "__main__":
    import time

    docking_cfg = DockingConfig(target_name='fa7', num_sub_proc=8)
    target = DockingVina(docking_cfg)

    smiles_list = ['O=C/C(=C/CC[C@H]([C@H]1CC[C@@]2([C@]1(C)CC[C@@]13[C@H]2CC[C@@H]2[C@]3(C1)CC[C@@H](C2(C)C)O)C)C)/C',
                   'Oc1ccc2c(c1)[C@@]1(C)CCN([C@@H](C2)[C@H]1C)CCc1ccccc1',
                   'CCCCCCCCCCCCCC[C@H]([C@H]([C@H](CO)N)O)O',
                   'O=C(O[C@H]1[C@H](Oc2ccc(cc2)c2cc(=O)c3c(o2)cc(cc3O)O)O[C@@H]([C@H]([C@@H]1O)O)COC(=O)/C=C/c1ccc(cc1)O)/C=C/c1ccc(cc1)O',
                   'C=CC#Cc1ccc(s1)c1cccs1',
                   'CC(=O)O[C@@H]1CC[C@H]2[C@@H](C1)N([C@H](C2)C(=N)O)C(=O)[C@H](N=C([C@H](Cc1ccc(cc1)O)O)O)Cc1ccc(cc1)O',
                   'C=C1[C@@H]2CC[C@@H]3[C@](C1=O)([C@@H]2O)[C@@]1(O)OC[C@@]23[C@@H](O)CCC([C@H]2[C@@H]1O)(C)C',
                   'COC(=O)[C@]1(C)CCC[C@]2([C@H]1CC=C1[C@@H]2CCC(=C1)C(C)C)C']

    st = time.time()
    affinity_list = target.predict(smiles_list)
    print(f'finish docking in {time.time() - st} seconds')

    for i, smi in enumerate(smiles_list):
        print(f'{smi}  ==> {affinity_list[i]}')

