from typing import Union, List

import torch
from pathlib import Path
from logging import getLogger

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from ml.common import TrainEpoch, Results

logger = getLogger('app')


class MySummaryWriter:
    def __init__(self, output_dir: Path):
        self._writer = SummaryWriter(log_dir=str(output_dir))

    def add_performance_epoch(self, train_epoch: TrainEpoch,
                              val_epoch: TrainEpoch, epoch_id: int):
        writer = self._writer
        for key, _ in train_epoch.metrics.items():
            writer.add_scalars(key, {'train': train_epoch.metrics[key]}, epoch_id)
            writer.add_scalars(key, {'validation': val_epoch.metrics[key]}, epoch_id)

    def add_model(self, model: torch.nn.Module, feature_batch: Union[Tensor, List[Tensor]]):
        self._writer.add_graph(model=model, input_to_model=feature_batch)

    def add_protein_results(self, protein_results: Results):
        pass
