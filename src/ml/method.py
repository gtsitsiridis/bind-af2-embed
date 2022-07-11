from __future__ import annotations

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
from logging import getLogger
from ml.template import MethodName, RunTemplate

logger = getLogger('app')


class Method(metaclass=ABCMeta):
    def __init__(self, template: RunTemplate, dataset: Dataset):
        self._template = template
        self._name = template.method_name
        self._dataset = dataset
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

        self._model = self._init_model()
        self._optimizer = self._init_optimizer()
        self._loss = self._init_loss()

    @staticmethod
    def get_method(template: RunTemplate, dataset: Dataset, max_length: int) -> Method:
        name = template.method_name

        if name == MethodName.EMBEDDINGS:
            return EmbeddingsMethod(dataset=dataset, template=template)
        elif name == MethodName.DISTMAPS:
            return DistMapsMethod(dataset=dataset, template=template, max_length=max_length)
        if name == MethodName.COMBINED_V1:
            return CombinedV1Method(dataset=dataset, template=template, max_length=max_length)
        elif name == MethodName.COMBINED_V2:
            return CombinedV2Method(dataset=dataset, template=template, max_length=max_length)
        else:
            assert False, f"The method {name.name} has not been defined yet."

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
    def _init_BCEWithLogits_loss(template: RunTemplate, device: str, max_length: int) -> _Loss:
        loss_params = template.loss_params
        pos_weights = torch.tensor(loss_params["pos_weights"]).to(device)
        assert len(loss_params['pos_weights']) == 4, \
            'the param pos_weights should have a length of 4'
        # weight is used to mute other ligands (e.g. 0,0,0,1 to test binding vs non-binding)
        weight = None
        if 'weight' in loss_params:
            assert len(loss_params['weight']) == 4, \
                'the param weights should have a length of 4'
            logger.warning("Custom weights have been passed to the BCE loss!")
            weight = torch.tensor(loss_params['weight']).to(device)
            weight = weight.expand(max_length, 4)
            weight = weight.t()

        pos_weights = pos_weights.expand(max_length, 4)
        pos_weights = pos_weights.t()

        # https://stackoverflow.com/questions/57021620/how-to-calculate-unbalanced-weights-for-bcewithlogitsloss-in-pytorch
        # https://discuss.pytorch.org/t/using-bcewithlogisloss-for-multi-label-classification/67011
        return torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weights, weight=weight)

    @staticmethod
    def _init_adamax_optimizer(template: RunTemplate, model: torch.nn.Module):
        optimizer_params = template.optimizer_params
        optim_args = {'lr': optimizer_params['lr'], 'betas': (0.9, 0.999), 'eps': optimizer_params['eps'],
                      'weight_decay': optimizer_params['weight_decay']}
        return torch.optim.Adamax(model.parameters(), **optim_args)


class EmbeddingsMethod(Method):
    def __init__(self, template: RunTemplate, dataset: Dataset):
        self._max_length = dataset.determine_max_length()
        self._embedding_size = dataset.embedding_size
        super().__init__(template, dataset=dataset)

    def _init_model(self) -> torch.nn.Module:
        model_params = self._template.model_params
        return models.EmbeddingsModel(model_params['features'], model_params['kernel'], model_params['dropout']).to(
            self.device)

    def _init_optimizer(self) -> Optimizer:
        return self._init_adamax_optimizer(template=self._template, model=self.model)

    def _init_loss(self) -> _Loss:
        return self._init_BCEWithLogits_loss(template=self._template, device=self.device, max_length=self._max_length)

    def get_dataset(self, ids: list, writer: MySummaryWriter = None) -> torch.utils.data.Dataset:
        return datasets.EmbeddingsDataset(ids, self._dataset)


class DistMapsMethod(Method):
    def __init__(self, template: RunTemplate, dataset: Dataset, max_length: int):
        self._max_length = max_length
        super().__init__(template, dataset=dataset)

    def _init_model(self) -> torch.nn.Module:
        model_params = self._template.model_params
        return models.DistMapsModel(max_length=self._max_length, feature_channels=model_params['features'],
                                    kernel_size=model_params['kernel'], depth=model_params['depth'],
                                    dropout=model_params['dropout']).to(self.device)

    def _init_optimizer(self) -> Optimizer:
        return self._init_adamax_optimizer(template=self._template, model=self.model)

    def _init_loss(self) -> _Loss:
        return self._init_BCEWithLogits_loss(template=self._template, device=self.device, max_length=self._max_length)

    def get_dataset(self, ids: list, writer: MySummaryWriter = None) -> torch.utils.data.Dataset:
        return datasets.DistMapsDataset(ids, self._dataset, writer=writer, max_length=self._max_length)


class CombinedV1Method(Method):
    def __init__(self, template: RunTemplate, dataset: Dataset, max_length: int):
        self._max_length = max_length
        self._embedding_size = dataset.embedding_size
        super().__init__(template, dataset=dataset)

    def _init_model(self) -> torch.nn.Module:
        model_params = self._template.model_params
        input_dimensions = 2 * self._max_length + self._embedding_size
        return models.CombinedModelV1(input_dimensions, model_params['features'], model_params['kernel'],
                                      model_params['dropout']).to(
            self.device)

    def _init_optimizer(self) -> Optimizer:
        return self._init_adamax_optimizer(template=self._template, model=self.model)

    def _init_loss(self) -> _Loss:
        return self._init_BCEWithLogits_loss(template=self._template, device=self.device, max_length=self._max_length)

    def get_dataset(self, ids: list, writer: MySummaryWriter = None) -> torch.utils.data.Dataset:
        return datasets.CombinedV1Dataset(ids, self._dataset, max_length=self._max_length,
                                          embedding_size=self._embedding_size)


class CombinedV2Method(Method):
    def __init__(self, template: RunTemplate, dataset: Dataset, max_length: int):
        self._max_length = max_length
        super().__init__(template, dataset=dataset)

    def _init_model(self) -> torch.nn.Module:
        model_params = self._template.model_params
        return models.CombinedModelV2(max_length=self._max_length, emb_feature_channels=model_params['emb_features'],
                                      distmap_depth=model_params['distmap_depth'],
                                      distmap_feature_channels=model_params['distmap_features'],
                                      kernel_size=model_params['kernel'], dropout=model_params['dropout']).to(
            self.device)

    def _init_optimizer(self) -> Optimizer:
        return self._init_adamax_optimizer(template=self._template, model=self.model)

    def _init_loss(self) -> _Loss:
        return self._init_BCEWithLogits_loss(template=self._template, device=self.device, max_length=self._max_length)

    def get_dataset(self, ids: list, writer: MySummaryWriter = None) -> torch.utils.data.Dataset:
        return datasets.CombinedV2Dataset(ids, self._dataset, writer=writer, max_length=self._max_length)
