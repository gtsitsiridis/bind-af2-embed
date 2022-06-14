from __future__ import annotations

import numpy as np
import h5py
import pandas as pd
from typing import Dict
from utils import FileManager
from pathlib import Path
from config import AppConfig


class Embedding(object):
    def __init__(self, tensor: np.array, prot_id: str):
        self._tensor = tensor
        self._prot_id = prot_id

    def reduce(self) -> np.array:
        return self._tensor.mean(axis=0)

    @property
    def tensor(self) -> np.array:
        return self._tensor

    @staticmethod
    def parse_file(embeddings_file) -> Dict[str, Embedding]:
        """
        Read embeddings from .h5-file
        :param embeddings_file:
        :return: dict with key: ID, value: per-residue embeddings
        """

        embeddings = dict()
        with h5py.File(embeddings_file, 'r') as f:
            for key, embedding in f.items():
                original_id = embedding.attrs['original_id']
                embeddings[original_id] = Embedding(tensor=np.array(embedding), prot_id=original_id)

        return embeddings


class Distogram(object):
    def __init__(self, tensor: np.array, prot_id: str):
        self._tensor = tensor
        self._prot_id = prot_id

    @property
    def tensor(self) -> np.array:
        return self._tensor

    @staticmethod
    def parse_files(distogram_dir) -> Dict[str, Distogram]:
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
            prot_id = file.name.split('.')[0]
            distograms[prot_id] = Distogram(tensor=np.load(str(file)), prot_id=prot_id)
        return distograms


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

    def to_list(self) -> list:
        return [''.join(x) for x in self._tensor.astype(str)]

    @property
    def tensor(self) -> np.array:
        return self._tensor

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


class ProteinData(object):
    def __init__(self, prot_id: str, sequence: Sequence, embedding: Embedding, bind_annotation: BindAnnotation,
                 distogram: Distogram):
        self._prot_id = prot_id
        self._bind_annotation = bind_annotation
        self._embedding = embedding
        self._sequence = sequence
        self._distogram = distogram

    @property
    def prot_id(self):
        return self.prot_id

    @property
    def bind_annotation(self):
        return self._bind_annotation

    @property
    def embedding(self):
        return self._embedding

    @property
    def sequence(self):
        return self._sequence

    @property
    def distogram(self):
        return self._distogram

    def __len__(self):
        return len(self._sequence)


class Dataset(object):

    def __init__(self, config: AppConfig):
        files = config.get_files()
        self._prot_ids, self._fold_array = FileManager.read_split_ids(files['splits'])
        sequences = Sequence.read_fasta(files['sequences'])
        bind_annotations = BindAnnotation.parse_files(files['biolip_annotations'], sequences=sequences)
        embeddings = Embedding.parse_file(files['embeddings'])
        distograms = Distogram.parse_files(files['distogram_dir'])

        self._proteins: Dict[str, ProteinData] = dict()
        for prot_id in sequences.keys():
            self._proteins[prot_id] = ProteinData(prot_id=prot_id, sequence=sequences[prot_id],
                                                  bind_annotation=bind_annotations[prot_id],
                                                  embedding=embeddings[prot_id],
                                                  distogram=distograms[prot_id])

    def get_proteins(self) -> Dict[str, ProteinData]:
        return self._proteins

    def determine_max_length(self):
        """Get maximum length in set of sequences"""
        _prot_ids = self._prot_ids
        proteins = self._proteins
        max_len = 0
        for i in _prot_ids:
            if len(proteins[i]) > max_len:
                max_len = len(proteins[i])

        return max_len

    @property
    def prot_ids(self) -> list:
        return self._prot_ids

    @property
    def fold_array(self) -> list:
        return self._fold_array

    def reduced_data(self, normalize: bool = False) -> (pd.DataFrame, np.array):
        """
        Reduces the dataset to protein level labels and embeddings by computing the mean values of their residues.
        :param normalize: should the label counts be normalized by the protein length?
        :return: dataframe containing reduced labels, np array containing reduced embeddings
        """
        # reduce embeddings and labels
        reduced_embeddings = []
        reduced_labels = []
        keys = []
        for key, protein in self._proteins.items():
            embedding = protein.embedding
            reduced_embeddings.append(embedding.reduce())
            keys.append(key)
            bind_annot = protein.bind_annotation
            reduced_labels.append(bind_annot.reduce(normalize=normalize))

        # reduced df
        df = pd.DataFrame(index=keys)
        df[BindAnnotation.names()] = reduced_labels
        df[list(map(lambda x: f'{x}_one', BindAnnotation.names()))] = list(map(lambda x: x > 0, reduced_labels))
        df['label'] = list(map(lambda x: int(np.argmax(x[0:3])), reduced_labels))
        df.label = df.label.apply(lambda label_id: BindAnnotation.id2name(label_id))

        return df, reduced_embeddings

    def long_data(self) -> (pd.DataFrame, np.array, np.array, np.array):
        """
        Combine all sequence info.

        :return:
        dataframe containing info for each residue in each protein,
        np array (M, 1024) containing associated embeddings,
        np array (M, 4) containing associated binding residue labels,
        np array (M, 2 * L_MAX) containing distograms,
        where M are the total residues and L_MAX the maximum protein length in the dataset
        """

        # sequences = self.__sequences
        # embeddings = self.__embeddings
        # distograms = self.__distograms
        proteins = self._proteins

        residues = []
        label_strs = []
        lengths = []
        positions = []
        protein_ids = []
        label_tensors = []
        embedding_tensors = []
        distogram_tensors = []
        max_distogram_length = 0
        for key, protein in proteins.items():
            distogram_tensor = protein.distogram.tensor
            seq = protein.sequence
            bind_annot = protein.bind_annotation
            distogram_tensor_2d = distogram_tensor.reshape(distogram_tensor.shape[0], -1)
            if distogram_tensor_2d.shape[0] != len(protein):
                print(f'Distogram length is different for id: {key}. '
                      f'Seq length: {str(len(protein))}, Distogram length: {str(distogram_tensor_2d.shape[0])}. '
                      f'Skipping...')
                continue

            embedding_tensor = protein.embedding.tensor
            if embedding_tensor.shape[0] != len(protein):
                print(f'Embedding length is different for id: {key}. '
                      f'Seq length: {str(len(protein))}, Embedding length: {str(embedding_tensor.shape[0])}. Skipping...')
                continue

            residues.extend(seq.to_list())
            label_strs.extend(bind_annot.to_list())
            lengths.extend([len(protein)] * len(protein))
            positions.extend(list(range(len(protein))))
            protein_ids.extend([key] * len(protein))
            label_tensors.append(bind_annot.tensor)
            embedding_tensors.append(embedding_tensor)
            if distogram_tensor_2d.shape[1] > max_distogram_length:
                max_distogram_length = distogram_tensor_2d.shape[1]
            distogram_tensors.append(distogram_tensor_2d)

        label_names = list(map(lambda x: bind_annot.ids2name(x), label_strs))

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
