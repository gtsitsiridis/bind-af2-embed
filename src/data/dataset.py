from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict
from utils import FileUtils
from config import AppConfig
from data.protein import Protein, ProteinStructure, BindAnnotation, Sequence, Embedding
from logging import getLogger
from random import sample

logger = getLogger('app')


class Dataset(object):

    def __init__(self, config: AppConfig, mode: str = 'all', subset: int = -1):
        """
        This object contains a dictionary of all protein objects and useful methods to manipulate them.
        :param config:
        :param subset: How many random proteins to include in the dataset.
        If subset = -1 then all the proteins are included.
        :param mode: one of ["train", "all", "test"]
        """
        files = config.get_input()

        assert mode in {'train', 'test', 'all'}, 'mode should be one of ["train", "all", "test"]'

        logger.info(f"Reading {mode} data")

        split_ids_files: list
        if mode in {'train', 'test'}:
            split_ids_files = files['splits'][mode]
        else:
            split_ids_files = files['splits']['train'] + files['splits']['test']

        logger.info("reading splits")
        self._prot_ids, self._fold_array = FileUtils.read_split_ids(split_ids_files, subset=subset)

        logger.info("reading protein data")
        sequences = Sequence.read_fasta(files['sequences'])
        bind_annotations = BindAnnotation.parse_files(files['biolip_annotations'], sequences=sequences)
        embeddings = Embedding.parse_file(files['embeddings'])
        structures = ProteinStructure.parse_files(distogram_dir=files['distogram_dir'], pdb_dir=files['pdb_dir'])
        self._embedding_size = list(embeddings.values())[0].tensor.shape[1]

        self._proteins: Dict[str, Protein] = dict()
        to_remove = []
        for prot_id in self._prot_ids:
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

            self._proteins[prot_id] = Protein(prot_id=prot_id, sequence=seq,
                                              bind_annotation=bind_annot,
                                              embedding=embed,
                                              structure=structure)
        for val in to_remove:
            idx = self._prot_ids.index(val)
            del self.prot_ids[idx]
            del self.fold_array[idx]

        assert len(self._prot_ids) == len(self.proteins), 'Something went wrong. We should not be here.'

        logger.info("Number of proteins: " + str(len(self)))

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

    def to_feature_tensor_dict(self) -> Dict[str, np.array]:
        feature_dict: Dict[str, np.array] = dict()
        max_length = self.determine_max_length()
        embedding_size = self._embedding_size

        # pad features based on max_length
        for prot_id, protein in self.proteins.items():
            tensor = protein.to_feature_tensor()
            feature_dict[prot_id] = np.pad(tensor,
                                           ([0, 0], [0, 2 * max_length + embedding_size - tensor.shape[1]]),
                                           mode='constant')

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
