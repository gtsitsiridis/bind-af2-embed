from __future__ import annotations

import numpy as np
import h5py
import pandas as pd
from typing import Dict
from utils import FileManager
from pathlib import Path
from config import AppConfig


class Embedding(object):
    def __init__(self, tensor: np.array, id: str):
        self.__tensor = tensor
        self.id = id

    def reduce(self) -> np.array:
        return self.__tensor.mean(axis=0)

    def get_tensor(self) -> np.array:
        return self.__tensor

    @staticmethod
    def read_embeddings(embeddings_file) -> Dict[str, Embedding]:
        """
        Read embeddings from .h5-file
        :param embeddings_file:
        :return: dict with key: ID, value: per-residue embeddings
        """

        embeddings = dict()
        with h5py.File(embeddings_file, 'r') as f:
            for key, embedding in f.items():
                original_id = embedding.attrs['original_id']
                embeddings[original_id] = Embedding(tensor=np.array(embedding), id=original_id)

        return embeddings


class Distogram(object):
    def __init__(self, tensor: np.array, id: str):
        self.__tensor = tensor
        self.id = id

    def get_tensor(self) -> np.array:
        return self.__tensor

    @staticmethod
    def read_distograms(distogram_dir) -> Dict[str, Distogram]:
        """
        Read distograms from directory containing npy files
        :param distogram_dir:
        :return: dict with key: ID, value: distogram
        """

        distogram_dir_path = Path(distogram_dir)
        assert distogram_dir_path.is_dir(), 'distogram_dir should be a directory'

        distograms = dict()
        for file in distogram_dir_path.iterdir():
            assert file.suffix == '.npy', f'invalid file found: {file.name}'
            id = file.name.split('.')[0]
            distograms[id] = Distogram(tensor=np.load(str(file)), id=id)
        return distograms


class Sequence(object):
    def __init__(self, sequence: str, id: str):
        self.__sequence = sequence
        self.__id = id
        self.__label_tensor = None

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
                        sequences[current_id] = Sequence(id=current_id, sequence=current_seq)
                    current_id = line[1:]
                    current_seq = ''
                else:
                    current_seq += line

        if current_seq is not None:
            sequences[current_id] = Sequence(id=current_id, sequence=current_seq)

        return sequences

    @staticmethod
    def labelId2Name(id: int) -> str:
        if id == 0:
            return 'metal'
        elif id == 1:
            return 'nuclear'
        elif id == 2:
            return 'small'
        else:
            return 'other'

    @staticmethod
    def labelIds2Name(id_str: str) -> str:
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
    def get_label_columns() -> list:
        return ['metal', 'nuclear', 'small', 'other']

    def get_label_tensor(self) -> np.array:
        return self.__label_tensor

    def get_label_strings(self) -> list:
        return [''.join(x) for x in self.__label_tensor.astype(str)]

    def set_label_tensor(self, label_tensor: np.array):
        self.__label_tensor = label_tensor

    def get_reduced_label_tensor(self, normalize: bool = False):
        n = self.__label_tensor.shape[1]
        result = np.zeros(n, dtype=float)

        for i in range(0, n):
            result[i] = float(np.sum(self.__label_tensor[:, i]))
            if normalize:
                result[i] /= len(self)

        return result

    def get_residues(self):
        return list(str(self.__sequence))

    def __len__(self):
        return len(self.__sequence)

    def __str__(self):
        return self.__sequence


class Dataset(object):

    def __init__(self, config: AppConfig):
        files = config.get_files()
        self.__sequences = Sequence.read_fasta(files['sequences'])
        self.__ids, self.__fold_array = FileManager.read_split_ids(files['splits'])
        self._set_labels(files['biolip_annotations'])
        self.__embeddings = Embedding.read_embeddings(files['embeddings'])
        self.__distograms = Distogram.read_distograms(files['distogram_dir'])

    @staticmethod
    def _get_zero_based_residues(residues):
        residues_0_ind = []
        for r in residues:
            residues_0_ind.append(int(r) - 1)

        return residues_0_ind

    def determine_max_length(self):
        """Get maximum length in set of sequences"""
        ids = self.__ids
        sequences = self.__sequences
        max_len = 0
        for i in ids:
            if len(sequences[i]) > max_len:
                max_len = len(sequences[i])

        return max_len

    def _set_labels(self, binding_residues_file_dict: Dict[str, str]):
        """
        Read binding residues for metal, nucleic acids, and small molecule binding

        :param binding_residues_file_dict:
        :return:
        """

        ids = self.__ids
        sequences = self.__sequences

        metal_residues = FileManager.read_binding_residues(binding_residues_file_dict["metal"])
        nuclear_residues = FileManager.read_binding_residues(binding_residues_file_dict["nuclear"])
        small_residues = FileManager.read_binding_residues(binding_residues_file_dict["small"])

        for prot_id in ids:
            sequence = sequences[prot_id]
            prot_length = len(sequence)
            binding_tensor = np.zeros([prot_length, 4], dtype=np.int32)

            metal_res = nuc_res = small_res = []

            if prot_id in metal_residues.keys():
                metal_res = metal_residues[prot_id]
            if prot_id in nuclear_residues.keys():
                nuc_res = nuclear_residues[prot_id]
            if prot_id in small_residues.keys():
                small_res = small_residues[prot_id]

            metal_residues_0_ind = self._get_zero_based_residues(metal_res)
            nuc_residues_0_ind = self._get_zero_based_residues(nuc_res)
            small_residues_0_ind = self._get_zero_based_residues(small_res)
            other_residues_0_ind = list(
                set(range(len(sequence))) - set(metal_residues_0_ind) - set(nuc_residues_0_ind) - set(
                    small_residues_0_ind))

            binding_tensor[metal_residues_0_ind, 0] = 1
            binding_tensor[nuc_residues_0_ind, 1] = 1
            binding_tensor[small_residues_0_ind, 2] = 1
            binding_tensor[other_residues_0_ind, 3] = 1

            sequence.set_label_tensor(binding_tensor)

    def get_labels(self) -> Dict[str, np.array]:
        sequences = self.__sequences
        return {key: seq.get_label_tensor() for key, seq in sequences.items()}

    def get_sequences(self) -> Dict[str, Sequence]:
        return self.__sequences

    def get_ids(self) -> list:
        return self.__ids

    def get_fold_array(self) -> list:
        return self.__fold_array

    def get_embeddings(self) -> Dict[str, Embedding]:
        return self.__embeddings

    def get_reduced_data(self, normalize: bool = False) -> (pd.DataFrame, np.array):
        """
        Reduces the dataset to protein level labels and embeddings by computing the mean values of their residues.
        :param normalize: should the label counts be normalized by the protein length?
        :return: dataframe containing reduced labels, np array containing reduced embeddings
        """
        # reduce embeddings and labels
        reduced_embeddings = []
        reduced_labels = []
        keys = []
        for key, embedding in self.__embeddings.items():
            reduced_embeddings.append(embedding.reduce())
            sequence = self.__sequences[key]
            keys.append(key)
            reduced_labels.append(sequence.get_reduced_label_tensor(normalize=normalize))

        # reduced df
        df = pd.DataFrame(index=keys)
        df[Sequence.get_label_columns()] = reduced_labels
        df[list(map(lambda x: f'{x}_one', Sequence.get_label_columns()))] = list(map(lambda x: x > 0, reduced_labels))
        df['label'] = list(map(lambda x: int(np.argmax(x[0:3])), reduced_labels))
        df.label = df.label.apply(lambda label_id: Sequence.labelId2Name(label_id))

        return df, reduced_embeddings

    def get_long_data(self) -> (pd.DataFrame, np.array, np.array, np.array):
        """
        Combine all sequence info.

        :return:
        dataframe containing info for each residue in each protein,
        np array (M, 1024) containing associated embeddings,
        np array (M, 4) containing associated binding residue labels,
        np array (M, 2 * L_MAX) containing distograms,
        where M are the total residues and L_MAX the maximum protein length in the dataset
        """

        sequences = self.__sequences
        embeddings = self.__embeddings
        distograms = self.__distograms

        residues = []
        label_strs = []
        lengths = []
        positions = []
        protein_ids = []
        label_tensors = []
        embedding_tensors = []
        distogram_tensors = []
        max_distogram_length = 0
        for key, seq in sequences.items():
            distogram_tensor = distograms[key].get_tensor()
            distogram_tensor_2d = distogram_tensor.reshape(distogram_tensor.shape[0], -1)
            if distogram_tensor_2d.shape[0] != len(seq):
                print(f'Distogram length is different for id: {key}. '
                      f'Seq length: {str(len(seq))}, Distogram length: {str(distogram_tensor_2d.shape[0])}. Skipping...')
                continue

            embedding_tensor = embeddings[key].get_tensor()
            if embedding_tensor.shape[0] != len(seq):
                print(f'Embedding length is different for id: {key}. '
                      f'Seq length: {str(len(seq))}, Embedding length: {str(embedding_tensor.shape[0])}. Skipping...')
                continue

            residues.extend(seq.get_residues())
            label_strs.extend(seq.get_label_strings())
            lengths.extend([len(seq)] * len(seq))
            positions.extend(list(range(len(seq))))
            protein_ids.extend([key] * len(seq))
            label_tensors.append(seq.get_label_tensor())
            embedding_tensors.append(embedding_tensor)
            if distogram_tensor_2d.shape[1] > max_distogram_length:
                max_distogram_length = distogram_tensor_2d.shape[1]
            distogram_tensors.append(distogram_tensor_2d)

        label_names = list(map(lambda x: Sequence.labelIds2Name(x), label_strs))

        df = pd.DataFrame(zip(positions, residues, lengths, protein_ids, label_strs, label_names),
                          columns=['position', 'residue', 'protein_length', 'protein_id', 'label_id', 'label_name'])
        label_tensors_array = np.concatenate(label_tensors)
        embedding_tensors_array = np.concatenate(embedding_tensors)
        # pad distograms
        for i in range(len(distogram_tensors)):
            distogram_tensor = distogram_tensors[i]
            distogram_tensors[i] = np.pad(distogram_tensor,
                                          ([0, 0], [0, max_distogram_length - distogram_tensor.shape[1]]),
                                          mode='constant')
        distogram_tensors_array = np.concatenate(distogram_tensors)
        return df, embedding_tensors_array, label_tensors_array, distogram_tensors_array
