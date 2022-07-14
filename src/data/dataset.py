from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict
from utils import FileUtils
from config import InputConfig
from data.protein import Protein, ProteinStructure, BindAnnotation, Sequence, Embedding
from logging import getLogger
import statistics

logger = getLogger('app')


class Dataset(object):
    """This object contains a dictionary of all protein objects and useful methods to manipulate them."""

    def __init__(self, proteins: Dict[str, Protein], fold_array: list, embedding_size: int):
        self._proteins = proteins
        self._fold_array = fold_array
        self._prot_ids = list(proteins.keys())
        self._embedding_size = embedding_size
        logger.info("Number of proteins: " + str(len(self)))

    @staticmethod
    def full_dataset(config: InputConfig) -> Dataset:
        """
        :param config:
        """
        files = config.files

        proteins: Dict[str, Protein]
        fold_array: list
        prot_ids: list
        embedding_size: int

        logger.info(f"Reading data")

        split_ids_files = files['splits']['train'] + files['splits']['test']

        logger.info("reading splits")
        prot_ids, fold_array = FileUtils.read_split_ids(split_ids_files)

        logger.info("reading protein data")
        sequences = Sequence.read_fasta(files['sequences'])
        bind_annotations = BindAnnotation.parse_files(files['biolip_annotations'], sequences=sequences)
        embeddings = Embedding.parse_file(files['embeddings'])
        structures = ProteinStructure.parse_files(distogram_dir=files['distogram_dir'], pdb_dir=files['pdb_dir'])
        embedding_size = list(embeddings.values())[0].tensor.shape[1]

        proteins = dict()

        to_remove = []
        splits_size_dict = dict()
        for idx, prot_id in enumerate(prot_ids):
            seq = sequences[prot_id]
            structure = structures[prot_id]
            bind_annot = bind_annotations[prot_id]
            embed = embeddings[prot_id]
            if structure.distogram_tensor.shape[0] != len(seq):
                logger.warning(f'Distogram length is different for id: {prot_id}. '
                               f'Seq length: {str(len(seq))}, '
                               f'Distogram length: {str(structure.distogram_tensor.shape[0])}. '
                               f'Skipping...')
                to_remove.append(prot_id)
                continue

            proteins[prot_id] = Protein(prot_id=prot_id, sequence=seq,
                                        bind_annotation=bind_annot,
                                        embedding=embed,
                                        structure=structure)
        for val in to_remove:
            idx = prot_ids.index(val)
            del prot_ids[idx]
            del fold_array[idx]

        assert len(prot_ids) == len(proteins), 'Something went wrong. We should not be here.'

        return Dataset(proteins=proteins, fold_array=fold_array, embedding_size=embedding_size)

    def get_subset(self, config: InputConfig, mode: str) -> Dataset:
        """
        :param config:
        :param mode: one of ["train", "all", "test"]
        :return:
        """
        assert mode in {'train', 'test'}, 'mode should be one of ["train", "test"]'
        max_length = config.params.get('max_length', None)
        min_length = config.params.get('min_length', None)
        subset = config.params.get('subset', None)
        plddt_limit = config.params.get('plddt_limit', None)

        files = config.files['splits'][mode]
        split_ids = [file['id'] for file in files]
        proteins: Dict[str, Protein] = dict()
        fold_array: list = list()
        embedding_size: int = self.embedding_size

        splits_size_dict = dict()
        for idx, prot_id in enumerate(self.prot_ids):
            protein = self.proteins[prot_id]

            fold = self.fold_array[idx]
            if fold not in split_ids:
                continue
            if plddt_limit is not None and np.mean(protein.structure.plddt_tensor) < plddt_limit:
                continue
            if max_length is not None and (0 < max_length < len(protein)):
                continue
            if min_length is not None and min_length > len(protein):
                continue
            if subset is not None and subset > 0:
                if fold in splits_size_dict.keys():
                    if splits_size_dict[fold] >= subset:
                        continue
                    splits_size_dict[fold] += 1
                else:
                    splits_size_dict[fold] = 1

            proteins[prot_id] = protein
            fold_array.append(fold)

        return Dataset(proteins=proteins, embedding_size=embedding_size, fold_array=fold_array)

    def __len__(self):
        return len(self.proteins)

    @property
    def proteins(self) -> Dict[str, Protein]:
        return self._proteins

    @property
    def prot_ids(self) -> list:
        return self._prot_ids

    @property
    def fold_array(self) -> list:
        return self._fold_array

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    def determine_max_length(self):
        """Get maximum length in set of sequences"""
        _prot_ids = self._prot_ids
        proteins = self._proteins
        max_len = 0
        for i in _prot_ids:
            if len(proteins[i]) > max_len:
                max_len = len(proteins[i])

        return max_len

    def to_feature_tensor_dict(self, max_length: int) -> Dict[str, np.array]:
        feature_dict: Dict[str, np.array] = dict()
        embedding_size = self._embedding_size

        # pad features based on max_length
        for prot_id, protein in self.proteins.items():
            tensor = protein.to_feature_tensor()
            padding = 2 * max_length + embedding_size - tensor.shape[1]
            if padding > 0:
                feature_dict[prot_id] = np.pad(tensor,
                                               ([0, 0], [0, padding]),
                                               mode='constant')
            else:
                feature_dict[prot_id] = tensor[:, :2 * max_length + embedding_size]

        return feature_dict

    def to_bind_annot_tensor_dict(self) -> Dict[str, np.array]:
        bind_annot_dict: Dict[str, np.array] = dict()
        for prot_id, protein in self.proteins.items():
            bind_annot_dict[prot_id] = protein.bind_annotation.tensor

        return bind_annot_dict

    def to_sequence_str_dict(self) -> Dict[str, str]:
        seq_dict: Dict[str, str] = dict()
        for prot_id, protein in self.proteins.items():
            seq_dict[prot_id] = str(protein.sequence)

        return seq_dict

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
        prot_lengths = []
        for key, protein in self._proteins.items():
            embedding = protein.embedding
            reduced_embeddings.append(embedding.reduce())
            keys.append(key)
            bind_annot = protein.bind_annotation
            reduced_labels.append(bind_annot.reduce(normalize=normalize))
            prot_lengths.append(len(protein))

        # reduced df
        df = pd.DataFrame(index=keys)
        df[BindAnnotation.names()] = reduced_labels
        df['protein_length'] = prot_lengths
        df[list(map(lambda x: f'{x}_one', BindAnnotation.names()))] = list(map(lambda x: x > 0, reduced_labels))
        df['label'] = list(map(lambda x: int(np.argmax(x[0:3])), reduced_labels))
        df.label = df.label.apply(lambda label_id: BindAnnotation.id2name(label_id))

        if normalize:
            df['other'] = df.apply(lambda x: 1 - x['nuclear'] - x['metal'] - x['small'], axis=1)
        else:
            df['other'] = df.apply(lambda x: x['protein_length'] - x['nuclear'] - x['metal'] - x['small'], axis=1)

        return df, reduced_embeddings

    def summary(self) -> dict:
        prot_length = [len(prot) for prot in self.proteins.values()]
        plddt = [prot.structure.plddt_tensor for prot in self.proteins.values()]
        plddt = np.concatenate(plddt)
        result = {
            "num_prot": len(self),
            "mean_prot_length": statistics.mean(prot_length),
            "min_prot_length": min(prot_length),
            "max_prot_length": max(prot_length),
            "mean_plddt": statistics.mean(plddt),
        }

        return result

    def long_data(self) -> (pd.DataFrame, Dict[str, np.array]):
        """
        Combine all sequence info.

        :return:
        dataframe containing info for each residue in each protein,

        Dictionary containing the following tensors:

        'embeddings' -> np array (M, 1024) containing associated embeddings,

        'binding_annotations' -> np array (M, 4) containing associated binding residue labels,

        'distograms' -> np array (M, 2 * L_MAX) containing distograms,

        where M are the total residues and L_MAX the maximum protein length in the dataset
        """

        proteins = self._proteins

        residues = []
        bind_annot_ids = []
        lengths = []
        positions = []
        protein_ids = []
        plddts = []
        bind_annot_tensors = []
        embedding_tensors = []
        distogram_tensors = []
        max_distogram_length = 0
        for key, protein in proteins.items():
            distogram_tensor_2d = protein.structure.distogram_tensor_2D()
            seq = protein.sequence
            bind_annot = protein.bind_annotation
            if distogram_tensor_2d.shape[0] != len(protein):
                logger.info(f'Distogram length is different for id: {key}. '
                            f'Seq length: {str(len(protein))}, Distogram length: {str(distogram_tensor_2d.shape[0])}. '
                            f'Skipping...')
                continue

            embedding_tensor = protein.embedding.tensor
            if embedding_tensor.shape[0] != len(protein):
                logger.info(f'Embedding length is different for id: {key}. '
                            f'Seq length: {str(len(protein))}, Embedding length: {str(embedding_tensor.shape[0])}. '
                            f'Skipping...')
                continue

            residues.extend(seq.to_list())
            bind_annot_ids.extend(bind_annot.to_ids())
            lengths.extend([len(protein)] * len(protein))
            positions.extend(list(range(len(protein))))
            protein_ids.extend([key] * len(protein))
            plddts.extend(protein.structure.plddt_tensor)
            bind_annot_tensors.append(bind_annot.tensor)
            embedding_tensors.append(embedding_tensor)
            if distogram_tensor_2d.shape[1] > max_distogram_length:
                max_distogram_length = distogram_tensor_2d.shape[1]
            distogram_tensors.append(distogram_tensor_2d)

        bind_annot_names = list(map(lambda x: bind_annot.ids2name(x), bind_annot_ids))

        df = pd.DataFrame(zip(positions, residues, lengths, protein_ids, bind_annot_ids, bind_annot_names, plddts),
                          columns=['position', 'residue', 'protein_length', 'protein_id', 'bind_annot_id',
                                   'bind_annot_name',
                                   'plddt'])
        bind_annot_tensors_merged = np.concatenate(bind_annot_tensors)
        embedding_tensors_merged = np.concatenate(embedding_tensors)
        # pad distograms
        for i in range(len(distogram_tensors)):
            distogram_tensor = distogram_tensors[i]
            distogram_tensors[i] = np.pad(distogram_tensor,
                                          ([0, 0], [0, max_distogram_length - distogram_tensor.shape[1]]),
                                          mode='constant')
        distogram_tensors_merged = np.concatenate(distogram_tensors)
        tensor_dict = {
            'embeddings': embedding_tensors_merged,
            'binding_annotations': bind_annot_tensors_merged,
            'distograms': distogram_tensors_merged
        }
        return df, tensor_dict
