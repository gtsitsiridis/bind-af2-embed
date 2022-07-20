from __future__ import annotations

import numpy as np
import pandas as pd
from plots import Plots
from umap import UMAP
from data.annotation import Sequence, BindAnnotation
from data.structure import ProteinStructure
from data.embedding import Embedding
from logging import getLogger

logger = getLogger('app')


class Protein(object):
    def __init__(self, prot_id: str, sequence: Sequence, embedding: Embedding, bind_annotation: BindAnnotation,
                 structure: ProteinStructure):
        self._prot_id = prot_id
        self._bind_annotation = bind_annotation
        self._embedding = embedding
        self._sequence = sequence
        self._structure = structure

    @property
    def prot_id(self):
        return self._prot_id

    @property
    def bind_annotation(self) -> BindAnnotation:
        return self._bind_annotation

    @property
    def embedding(self) -> Embedding:
        return self._embedding

    @property
    def sequence(self) -> Sequence:
        return self._sequence

    @property
    def structure(self) -> ProteinStructure:
        return self._structure

    def to_feature_tensor(self) -> np.array:
        return np.concatenate([self.embedding.tensor, self.structure.distogram_tensor_2D()], axis=1)

    def show_structure(self, show_mainchains: bool = False, show_sidechains: bool = False, color='lDDT',
                       ri_tensor: np.array = None):
        Plots.show_pdb(pdb_file=self.structure.pdb_file, show_mainchains=show_mainchains,
                       bind_annot_names=self.bind_annotation.to_names(),
                       show_sidechains=show_sidechains,
                       color=color, ri_tensor=ri_tensor)
        if color == 'lDDT':
            Plots.plot_plddt_legend()
        if color == 'ligand':
            Plots.plot_ligand_legend()
        if color == 'ri':
            Plots.plot_ri_legend()

    def show_umap(self, n_neighbors: int = 30, color: str = 'ligand', write: bool = False):
        distogram_tensor = self.structure.distogram_tensor_2D()
        embedding_tensor = self.embedding.tensor

        assert distogram_tensor.shape[0] == len(self), 'invalid distogram in the dataset'

        X = np.concatenate([distogram_tensor, embedding_tensor], axis=1)
        umap_2d = UMAP(n_neighbors=n_neighbors, n_components=2, init='random', random_state=42, verbose=True)
        umap_proj_2d = umap_2d.fit_transform(X)

        possible_colors = {'ligand', 'residue'}
        assert color in possible_colors, f'Invalid color given. Choose one of {str(possible_colors)}'
        color_vector = None
        color_title = None
        if color == 'ligand':
            color_title = 'Ligand'
            color_vector = self.bind_annotation.to_names()
        elif color == 'residue':
            color_title = 'Amino acid'
            color_vector = self.sequence.to_list()

        Plots.umap_2d(umap_proj_2d=umap_proj_2d,
                      data_title=self.prot_id,
                      color_title=color_title,
                      color=pd.Series((v for v in color_vector)),
                      hover_data={'ligand': self.bind_annotation.to_names(),
                                  'position': list(range(len(self))),
                                  'residue': self.sequence.to_list()},
                      n_neighbors_title=str(n_neighbors),
                      subset_title=str(len(self)),
                      write=write,
                      show=True)

    def __len__(self):
        return len(self._sequence)
