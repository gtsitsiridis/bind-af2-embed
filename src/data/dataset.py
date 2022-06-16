from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict
from utils import FileManager
from config import AppConfig
from data.protein import Protein, ProteinStructure, Sequence, BindAnnotation, Embedding


class Dataset(object):

    def __init__(self, config: AppConfig):
        files = config.get_files()
        self._prot_ids, self._fold_array = FileManager.read_split_ids(files['splits'])
        sequences = Sequence.read_fasta(files['sequences'])
        bind_annotations = BindAnnotation.parse_files(files['biolip_annotations'], sequences=sequences)
        embeddings = Embedding.parse_file(files['embeddings'])
        structures = ProteinStructure.parse_files(distogram_dir=files['distogram_dir'], pdb_dir=files['pdb_dir'])

        self._proteins: Dict[str, Protein] = dict()
        for prot_id in sequences.keys():
            self._proteins[prot_id] = Protein(prot_id=prot_id, sequence=sequences[prot_id],
                                              bind_annotation=bind_annotations[prot_id],
                                              embedding=embeddings[prot_id],
                                              structure=structures[prot_id])

    @property
    def proteins(self) -> Dict[str, Protein]:
        return self._proteins

    @property
    def prot_ids(self) -> list:
        return self._prot_ids

    @property
    def fold_array(self) -> list:
        return self._fold_array

    def determine_max_length(self):
        """Get maximum length in set of sequences"""
        _prot_ids = self._prot_ids
        proteins = self._proteins
        max_len = 0
        for i in _prot_ids:
            if len(proteins[i]) > max_len:
                max_len = len(proteins[i])

        return max_len

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
                print(f'Distogram length is different for id: {key}. '
                      f'Seq length: {str(len(protein))}, Distogram length: {str(distogram_tensor_2d.shape[0])}. '
                      f'Skipping...')
                continue

            embedding_tensor = protein.embedding.tensor
            if embedding_tensor.shape[0] != len(protein):
                print(f'Embedding length is different for id: {key}. '
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
