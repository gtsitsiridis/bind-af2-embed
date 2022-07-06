from enum import Enum

from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from data.dataset import Dataset

import ml.models as models
import ml.datasets as datasets
import torch
from abc import ABCMeta, abstractmethod
from typing import List, Union
from ml.summary_writer import MySummaryWriter


class MethodName(Enum):
    CNN1D_ALL = "CNN1D_ALL"
    CNN1D_EMBEDDINGS = "CNN1D_EMBEDDINGS"
    CNN2D_DISTMAPS = "CNN2D_DISTMAPS"
    CNN_COMBINED = "CNN_COMBINED"


class Method(metaclass=ABCMeta):
    def __init__(self, ml_config: dict, name: MethodName, dataset: Dataset):
        assert name.name in ml_config["methods"], f'The method {name.name} has not been configured.'
        model_config = ml_config["methods"][name.name]
        self._params = model_config['params']
        self._name = name
        self._dataset = dataset
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
    def model(self):
        return self._model

    @property
    def optimizer(self):
        return self._optimizer

    @model.setter
    def model(self, model: torch.nn.Module):
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

    def loss(self, pred_batch: Tensor, target_batch: Tensor, loss_mask_batch: Tensor) -> Tensor:
        loss_el_batch = self._loss(pred_batch, target_batch)  # loss_el_batch.shape=(B, 3, T)
        loss_el_masked_batch = loss_el_batch * loss_mask_batch  # loss_el_masked_batch.shape=(B, 3, T)
        return loss_el_masked_batch

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
    def get_dataset(ids: list, writer: MySummaryWriter = None) -> torch.utils.data.Dataset:
        pass

    @staticmethod
    def _init_BCEWithLogits_loss(params: dict, device: str, max_length: int) -> _Loss:
        pos_weights = torch.tensor(params["pos_weights"]).to(device)
        assert len(params['pos_weights']) == 4, \
            'the param pos_weights should have a length of 4'
        # weight is used to mute other ligands (e.g. 0,0,0,1 to test binding vs non-binding)
        weight = None
        if 'weight' in params:
            assert len(params['weight']) == 4, \
                'the param weights should have a length of 4'
            weight = torch.tensor(params['weight']).to(device)
            weight = weight.expand(max_length, 4)
            weight = weight.t()

        pos_weights = pos_weights.expand(max_length, 4)
        pos_weights = pos_weights.t()

        # https://stackoverflow.com/questions/57021620/how-to-calculate-unbalanced-weights-for-bcewithlogitsloss-in-pytorch
        # https://discuss.pytorch.org/t/using-bcewithlogisloss-for-multi-label-classification/67011
        return torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weights, weight=weight)

    @staticmethod
    def _init_adamax_optimizer(params: dict, model: torch.nn.Module):
        optim_args = {'lr': params['lr'], 'betas': (0.9, 0.999), 'eps': params['eps'],
                      'weight_decay': params['weight_decay']}
        return torch.optim.Adamax(model.parameters(), **optim_args)


class CNN1DAllMethod(Method):
    def __init__(self, ml_config: dict, dataset: Dataset):
        self._embedding_size = dataset.embedding_size
        super().__init__(ml_config, MethodName.CNN1D_ALL, dataset=dataset)

    def _init_model(self) -> torch.nn.Module:
        params = self._params
        input_dimensions = 2 * self._params['max_length'] + self._embedding_size
        return models.CNN1DModel(input_dimensions, params['features'], params['kernel'], params['dropout']).to(
            self.device)

    def _init_optimizer(self) -> Optimizer:
        return self._init_adamax_optimizer(params=self._params, model=self.model)

    def _init_loss(self) -> _Loss:
        return self._init_BCEWithLogits_loss(params=self._params, device=self.device,
                                             max_length=self._params['max_length'])

    def get_dataset(self, ids: list, writer: MySummaryWriter = None) -> torch.utils.data.Dataset:
        return datasets.CNN1DAllDataset(ids, self._dataset, max_length=self._params['max_length'],
                                        embedding_size=self._embedding_size)


class CNN1DEmbeddingsMethod(Method):
    def __init__(self, ml_config: dict, dataset: Dataset):
        self._max_length = dataset.determine_max_length()
        self._embedding_size = dataset.embedding_size
        super().__init__(ml_config, MethodName.CNN1D_EMBEDDINGS, dataset=dataset)

    def _init_model(self) -> torch.nn.Module:
        params = self._params
        input_dimensions = self._embedding_size
        return models.CNN1DModel(input_dimensions, params['features'], params['kernel'], params['dropout']).to(
            self.device)

    def _init_optimizer(self) -> Optimizer:
        return self._init_adamax_optimizer(params=self._params, model=self.model)

    def _init_loss(self) -> _Loss:
        return self._init_BCEWithLogits_loss(params=self._params, device=self.device, max_length=self._max_length)

    def get_dataset(self, ids: list, writer: MySummaryWriter = None) -> torch.utils.data.Dataset:
        return datasets.CNN1DEmbeddingsDataset(ids, self._dataset)


class CNN2DDistMapMethod(Method):
    def __init__(self, ml_config: dict, dataset: Dataset, max_length: int):
        self._max_length = max_length
        super().__init__(ml_config, MethodName.CNN2D_DISTMAPS, dataset=dataset)

    def _init_model(self) -> torch.nn.Module:
        return models.CNN2DModel(max_length=self._max_length).to(self.device)

    def _init_optimizer(self) -> Optimizer:
        return self._init_adamax_optimizer(params=self._params, model=self.model)

    def _init_loss(self) -> _Loss:
        return self._init_BCEWithLogits_loss(params=self._params, device=self.device, max_length=self._max_length)

    def get_dataset(self, ids: list, writer: MySummaryWriter = None) -> torch.utils.data.Dataset:
        return datasets.CNN2DDistMaps(ids, self._dataset, writer=writer, max_length=self._max_length)


class CNNCombinedMethod(Method):
    def __init__(self, ml_config: dict, dataset: Dataset, max_length: int):
        self._max_length = max_length
        super().__init__(ml_config, MethodName.CNN_COMBINED, dataset=dataset)

    def _init_model(self) -> torch.nn.Module:
        return models.CNNCombinedModel(max_length=self._max_length).to(self.device)

    def _init_optimizer(self) -> Optimizer:
        return self._init_adamax_optimizer(params=self._params, model=self.model)

    def _init_loss(self) -> _Loss:
        return self._init_BCEWithLogits_loss(params=self._params, device=self.device, max_length=self._max_length)

    def get_dataset(self, ids: list, writer: MySummaryWriter = None) -> torch.utils.data.Dataset:
        return datasets.CNNCombinedDataset(ids, self._dataset, writer=writer, max_length=self._max_length)
