from __future__ import annotations

import numpy as np
import h5py
from typing import Dict


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
