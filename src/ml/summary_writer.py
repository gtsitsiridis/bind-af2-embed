from typing import Union, List

import pandas as pd
import torch
from pathlib import Path
from logging import getLogger

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from ml.common import PerformanceMap, Results, Performance
from plots import Plots
import matplotlib.pyplot as plt
import numpy as np
from data.annotation import BindAnnotation

logger = getLogger('app')


class MySummaryWriter:
    def __init__(self, output_dir: Path):
        self._writer = SummaryWriter(log_dir=str(output_dir))

    def add_performance(self, performance: Performance, epoch_id: int = None):
        writer = self._writer
        for key in performance.keys():
            writer.add_scalar(key, performance[key], epoch_id)

    def add_model(self, model: torch.nn.Module, feature_batch: Union[Tensor, List[Tensor]]):
        self._writer.add_graph(model=model, input_to_model=feature_batch)

    def add_protein_results(self, protein_results: Results, cutoff: float, epoch: int = None):
        cf_figures = []
        roc_figures = []
        ri_figures = []

        def append_plots(class_label, df):
            cf_fig, cf_ax = plt.subplots()
            roc_fig, roc_ax = plt.subplots()
            ri_fig, ri_ax = plt.subplots()
            Plots.plot_confusion_matrix(y_true=df['target'], y_pred=df['prediction'],
                                        class_label=class_label, ax=cf_ax)
            Plots.plot_roc(y_true=df['target'], y_score=df['prob'], class_label=class_label, ax=roc_ax)
            Plots.plot_ri_hist(df.ri, ri_ax, class_label=class_label)
            cf_figures.append(cf_fig)
            roc_figures.append(roc_fig)
            ri_figures.append(ri_fig)

        # total predictions matrix
        df_total = protein_results.to_df(cutoff=cutoff)
        class_label_total = 'total'
        append_plots(class_label_total, df_total)

        # total predictions per ligand
        ligands = BindAnnotation.names()
        for ligand in ligands:
            df_subset = df_total[df_total.ligand == ligand]
            append_plots(ligand, df_subset)

        self._writer.add_figure(tag='roc',
                                figure=roc_figures,
                                global_step=epoch)
        self._writer.add_figure(tag='confusion matrix',
                                figure=cf_figures,
                                global_step=epoch)
        self._writer.add_figure(tag='RI distribution',
                                figure=ri_figures,
                                global_step=epoch)

    def add_performance_per_cutoff(self, protein_results: Results, cutoffs: List[float], epoch: int = None):
        perf_per_cutoff = protein_results.get_performance_per_cutoff(cutoffs=cutoffs)
        perf_per_cutoff = {cutoff: perf.to_dict() for cutoff, perf in perf_per_cutoff.items()}

        ligands = BindAnnotation.names()
        figures = []
        for class_label in ['total'] + ligands:
            fig, ax = plt.subplots()
            Plots.plot_performance_per_cutoff(metrics_per_cutoff=perf_per_cutoff,
                                              class_label=class_label,
                                              ax=ax)
            figures.append(fig)
        self._writer.add_figure(tag='performance_per_cutoff',
                                figure=figures,
                                global_step=epoch)
