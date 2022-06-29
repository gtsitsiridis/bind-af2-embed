from enum import Enum

from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from data.dataset import Dataset

from ml.models import CNN2Layers
from ml.datasets import CNN1DDataset, CNN2DDataset
import torch
from abc import ABCMeta, abstractmethod
from ml.summary_writer import MySummaryWriter
from ml.common import TrainEpoch
from typing import List, Union


class MethodName(Enum):
    CNN1D = "CNN1D"


class Method(metaclass=ABCMeta):
    def __init__(self, ml_config: dict, name: MethodName):
        assert name.name in ml_config["methods"], f'The method {name.name} has not been configured.'
        model_config = ml_config["methods"][name.name]
        self._params = model_config['params']
        self._name = name
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

        self._model = self._init_model()
        self._optimizer = self._init_optimizer()
        self._loss = self._init_loss()

    @property
    def name(self):
        return self._name

    @property
    def loss(self):
        return self._loss

    @property
    def model(self):
        return self._model

    @property
    def optimizer(self):
        return self._optimizer

    def reset(self):
        self._model = self._init_model()
        self._optimizer = self._init_optimizer()
        self._loss = self._init_loss()

    @model.setter
    def model(self, model: torch.nn.Module):
        print("setter of x called")
        self._model = model

    def forward(self, feature_batch: Union[Tensor, List[Tensor]]) -> Tensor:
        # feature_batch.shape=(B, 2*T + 1024, T)
        if type(feature_batch) is list:
            for i, feature_batch_i in enumerate(feature_batch):
                feature_batch[i] = feature_batch_i.to(self.device)
            pred_batch = self.model.forward(*feature_batch)  # pred_batch.shape=(B, 3, T)
        else:
            feature_batch = feature_batch.to(self.device)
            pred_batch = self.model.forward(feature_batch)

        return pred_batch

    @abstractmethod
    def _init_model(self) -> torch.nn.Module:
        pass

    @abstractmethod
    def _init_optimizer(self) -> Optimizer:
        pass

    @abstractmethod
    def _init_loss(self) -> _Loss:
        pass

    @staticmethod
    @abstractmethod
    def get_dataset(ids: list, dataset: Dataset) -> torch.utils.data.Dataset:
        pass


class CNN1DMethod(Method):
    def __init__(self, ml_config: dict, max_length: int, embedding_size: int):
        self._max_length = max_length
        self._embedding_size = embedding_size
        super().__init__(ml_config, MethodName.CNN1D)

    def _init_model(self) -> torch.nn.Module:
        params = self._params

        input_dimensions = 2 * self._max_length + self._embedding_size
        return CNN2Layers(input_dimensions, params['features'], params['kernel'], params['dropout']).to(self.device)

    def _init_optimizer(self) -> Optimizer:
        params = self._params
        model = self.model
        optim_args = {'lr': params['lr'], 'betas': (0.9, 0.999), 'eps': params['eps'],
                      'weight_decay': params['weight_decay']}
        return torch.optim.Adamax(model.parameters(), **optim_args)

    def _init_loss(self) -> _Loss:
        params = self._params
        pos_weights = torch.tensor(params["pos_weights"]).to(self.device)
        pos_weights = pos_weights.expand(self._max_length, 3)
        pos_weights = pos_weights.t()

        # https://stackoverflow.com/questions/57021620/how-to-calculate-unbalanced-weights-for-bcewithlogitsloss-in-pytorch
        # https://discuss.pytorch.org/t/using-bcewithlogisloss-for-multi-label-classification/67011
        return torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weights)

    @staticmethod
    def get_dataset(ids: list, dataset: Dataset) -> torch.utils.data.Dataset:
        return CNN1DDataset(ids, dataset)
