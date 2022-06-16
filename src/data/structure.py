from __future__ import annotations

import numpy as np
from typing import Dict
from pathlib import Path
from logging import getLogger

logger = getLogger('app')


class ProteinStructure(object):
    def __init__(self, distogram_tensor: np.array, pdb_file: str,
                 plddt_tensor: np.array, prot_id: str):
        self._pdb_file = pdb_file
        self._plddt_tensor = plddt_tensor
        self._distogram_tensor = distogram_tensor
        self._prot_id = prot_id

    @property
    def distogram_tensor(self) -> np.array:
        return self._distogram_tensor

    @property
    def pdb_file(self) -> str:
        return self._pdb_file

    @property
    def plddt_tensor(self) -> np.array:
        return self._plddt_tensor

    @property
    def prot_id(self) -> str:
        return self._prot_id

    def distogram_tensor_2D(self) -> np.array:
        return self.distogram_tensor.reshape(self.distogram_tensor.shape[0], -1)

    def avg_plddt(self) -> float:
        return float(np.mean(self._plddt_tensor))

    @staticmethod
    def parse_files(distogram_dir: str, pdb_dir: str) -> Dict[str, ProteinStructure]:
        """
        Parse protein structure related files.
        :param pdb_dir: directory containing .pdb files
        :param distogram_dir: directory containing npy files
        :return: dict with key: ID, value: ProteinStructure
        """

        distogram_dir_path = Path(distogram_dir)
        pdb_dir_path = Path(pdb_dir)
        assert distogram_dir_path.is_dir(), 'distogram_dir should be a directory'
        assert pdb_dir_path.is_dir(), 'pdb_dir should be a directory'

        distograms: Dict[str, np.array] = dict()
        pdb_files: Dict[str, str] = dict()
        plddts: Dict[str, np.array] = dict()
        for file in distogram_dir_path.iterdir():
            assert file.suffix == '.npy', f'invalid file found: {file.name}'
            prot_id = file.name.split('.')[0]
            distograms[prot_id] = np.load(str(file))

        for file in pdb_dir_path.iterdir():
            assert file.suffix == '.pdb', f'invalid file found: {file.name}'
            prot_id = file.name.split('_')[0]
            pdb_files[prot_id] = str(file)

            with open(str(file), "r") as f:
                lines = f.readlines()[1:-3]
                plddt_tensor = np.array(list(map(lambda line: float(line.split()[-2]), lines)))
                plddts[prot_id] = plddt_tensor

        structures: Dict[str, ProteinStructure] = dict()
        for prot_id in distograms.keys():
            structures[prot_id] = ProteinStructure(distogram_tensor=distograms[prot_id],
                                                   pdb_file=pdb_files[prot_id],
                                                   plddt_tensor=plddts[prot_id],
                                                   prot_id=prot_id)

        return structures
