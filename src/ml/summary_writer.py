from typing import Union, List

import pandas as pd
import torch
from pathlib import Path
from logging import getLogger

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from ml.common import Performance, Results, SinglePerformance
from plots import Plots
import matplotlib.pyplot as plt

logger = getLogger('app')


class MySummaryWriter:
    def __init__(self, output_dir: Path):
        self._writer = SummaryWriter(log_dir=str(output_dir))

    def add_single_performance(self, train_performance: SinglePerformance,
                               val_performance: SinglePerformance, epoch_id: int = None):
        writer = self._writer
        for key in train_performance.keys():
            writer.add_scalar(key + "_train", train_performance[key], epoch_id)
            writer.add_scalar(key + "_validation", val_performance[key], epoch_id)

    def add_model(self, model: torch.nn.Module, feature_batch: Union[Tensor, List[Tensor]]):
        self._writer.add_graph(model=model, input_to_model=feature_batch)

    def add_protein_results(self, protein_results: Results, cutoff: float, epoch: int = None):
        cf_figures = []
        roc_figures = []

        def append_plots(class_label, df):
            cf_fig, cf_ax = plt.subplots()
            roc_fig, roc_ax = plt.subplots()
            Plots.plot_confusion_matrix(y_true=df['target'], y_pred=df['prediction'],
                                        class_label=class_label, ax=cf_ax)
            Plots.plot_roc(y_true=df['target'], y_score=df['prob'], class_label=class_label, ax=roc_ax)

            cf_figures.append(cf_fig)
            roc_figures.append(roc_fig)

        # total predictions matrix
        df_total = protein_results.to_df(cutoff=cutoff)
        class_label_total = 'total'
        append_plots(class_label_total, df_total)

        # total predictions per ligand
        ligands = df_total.ligand.unique()
        for ligand in ligands:
            df_subset = df_total[df_total.ligand == ligand]
            append_plots(ligand, df_subset)

        self._writer.add_figure(tag='roc',
                                figure=roc_figures,
                                global_step=epoch)
        self._writer.add_figure(tag='confusion matrix',
                                figure=cf_figures,
                                global_step=epoch)
