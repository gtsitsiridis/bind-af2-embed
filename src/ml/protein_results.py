import numpy as np


class ProteinResults(object):

    def __init__(self, name: str, target: np.array, predictions: np.array, bind_cutoff=0.5):
        self.name = name
        self.target = target
        self.predictions = predictions
        self.bind_cutoff = bind_cutoff

