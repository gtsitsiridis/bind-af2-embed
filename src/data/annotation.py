from __future__ import annotations

import numpy as np
from typing import Dict
from utils import FileManager


class Sequence(object):
    def __init__(self, sequence: str, prot_id: str):
        self._sequence = sequence
        self._prot_id = prot_id

    @staticmethod
    def read_fasta(file_in) -> Dict[str, Sequence]:
        """
        Read sequences from FASTA file
        :param file_in:
        :return: dict with key: ID, value: sequence
        """
        sequences = dict()
        current_id = None
        current_seq = None

        with open(file_in) as read_in:
            for line in read_in:
                line = line.strip()
                if line.startswith(">"):
                    if current_seq is not None:
                        sequences[current_id] = Sequence(prot_id=current_id, sequence=current_seq)
                    current_id = line[1:]
                    current_seq = ''
                else:
                    current_seq += line

        if current_seq is not None:
            sequences[current_id] = Sequence(prot_id=current_id, sequence=current_seq)

        return sequences

    def to_list(self) -> list:
        return list(str(self._sequence))

    @property
    def prot_id(self) -> str:
        return self._prot_id

    def __len__(self):
        return len(self._sequence)

    def __str__(self):
        return self._sequence


class BindAnnotation(object):
    def __init__(self, tensor: np.array, prot_id: str):
        self._tensor = tensor
        self._prot_id = prot_id

    def reduce(self, normalize: bool = False) -> np.array:
        n = self._tensor.shape[1]
        result = np.zeros(n, dtype=float)

        for i in range(0, n):
            result[i] = float(np.sum(self._tensor[:, i]))
            if normalize:
                result[i] /= len(self)

        return result

    def to_ids(self) -> list:
        return [''.join(x) for x in self._tensor.astype(str)]

    @property
    def tensor(self) -> np.array:
        return self._tensor

    def to_names(self) -> list:
        return list(map(lambda x: self.ids2name(x), self.to_ids()))

    def __len__(self):
        return len(self._tensor)

    @staticmethod
    def parse_files(binding_residues_file_dict: Dict[str, str], sequences: Dict[str, Sequence]) -> \
            Dict[str, BindAnnotation]:
        """
        Read binding residues for metal, nucleic acids, and small molecule binding

        :param sequences:
        :param binding_residues_file_dict:
        :return:
        """

        metal_residues = FileManager.read_binding_residues(binding_residues_file_dict["metal"])
        nuclear_residues = FileManager.read_binding_residues(binding_residues_file_dict["nuclear"])
        small_residues = FileManager.read_binding_residues(binding_residues_file_dict["small"])

        bind_annotations: Dict[str, BindAnnotation] = dict()
        for prot_id, sequence in sequences.items():
            prot_length = len(sequence)
            binding_tensor = np.zeros([prot_length, 4], dtype=np.int32)

            metal_res = nuc_res = small_res = []

            if prot_id in metal_residues.keys():
                metal_res = metal_residues[prot_id]
            if prot_id in nuclear_residues.keys():
                nuc_res = nuclear_residues[prot_id]
            if prot_id in small_residues.keys():
                small_res = small_residues[prot_id]

            metal_residues_0_ind = BindAnnotation._get_zero_based_residues(metal_res)
            nuc_residues_0_ind = BindAnnotation._get_zero_based_residues(nuc_res)
            small_residues_0_ind = BindAnnotation._get_zero_based_residues(small_res)
            other_residues_0_ind = list(
                set(range(len(sequence))) - set(metal_residues_0_ind) - set(nuc_residues_0_ind) - set(
                    small_residues_0_ind))

            binding_tensor[metal_residues_0_ind, 0] = 1
            binding_tensor[nuc_residues_0_ind, 1] = 1
            binding_tensor[small_residues_0_ind, 2] = 1
            binding_tensor[other_residues_0_ind, 3] = 1

            bind_annotations[prot_id] = BindAnnotation(tensor=binding_tensor, prot_id=prot_id)
        return bind_annotations

    @staticmethod
    def _get_zero_based_residues(residues):
        residues_0_ind = []
        for r in residues:
            residues_0_ind.append(int(r) - 1)

        return residues_0_ind

    @staticmethod
    def id2name(id_: int) -> str:
        if id_ == 0:
            return 'metal'
        elif id_ == 1:
            return 'nuclear'
        elif id_ == 2:
            return 'small'
        else:
            return 'other'

    @staticmethod
    def ids2name(id_str: str) -> str:
        assert len(id_str) == 4, 'invalid input, expected 4 char string'
        res = []
        if id_str[0] == '1':
            res.append('metal')
        if id_str[1] == '1':
            res.append('nuclear')
        if id_str[2] == '1':
            res.append('small')
        if id_str[3] == '1':
            res.append('other')
        return ','.join(res)

    @staticmethod
    def names() -> list:
        return ['metal', 'nuclear', 'small', 'other']
