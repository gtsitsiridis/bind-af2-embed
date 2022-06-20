from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from data.dataset import Dataset

from ml.models import CNN2Layers
from ml.datasets import CNN1DDataset, CNN2DDataset
import torch
from abc import ABCMeta, abstractmethod


class Method(metaclass=ABCMeta):
    def __init__(self, ml_config: dict, name: str, dataset: Dataset):
        assert name in ml_config["models"], f'The method {name} has not been configured.'
        model_config = ml_config["models"][name]
        self._params = model_config['params']
        self._dataset = dataset
        self._name = name
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

    @property
    def name(self):
        return self._name

    @abstractmethod
    def init_model(self) -> torch.nn.Module:
        pass

    @abstractmethod
    def init_optimizer(self, model: torch.nn.Module) -> Optimizer:
        pass

    @abstractmethod
    def init_loss(self) -> _Loss:
        pass

    @staticmethod
    @abstractmethod
    def get_dataset(ids: list, dataset: Dataset) -> torch.utils.data.Dataset:
        pass


class CNN1DMethod(Method):
    def __init__(self, ml_config: dict, dataset: Dataset, max_length: int):
        super().__init__(ml_config, "CNN1D", dataset)
        self._max_length = max_length

    def init_model(self) -> torch.nn.Module:
        params = self._params

        input_dimensions = 2 * self._max_length + 1025
        return CNN2Layers(input_dimensions, params['features'], params['kernel'], params['dropout']).to(self.device)

    def init_optimizer(self, model: torch.nn.Module) -> Optimizer:
        params = self._params
        optim_args = {'lr': params['lr'], 'betas': (0.9, 0.999), 'eps': params['eps'],
                      'weight_decay': params['weight_decay']}
        return torch.optim.Adamax(model.parameters(), **optim_args)

    def init_loss(self) -> _Loss:
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
