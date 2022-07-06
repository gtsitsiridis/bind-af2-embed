from typing import List, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import py3Dmol
import matplotlib.pyplot as plt
from logging import getLogger
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

logger = getLogger('app')
BIND_ANNOT_COLORS = {'other': 'black',
                     'metal': 'orange',
                     'small': 'green',
                     'nuclear': 'red'}


class Plots(object):

    @staticmethod
    def umap_2d(umap_proj_2d: np.array, color: pd.Series,
                hover_data: Dict[str, any],
                n_neighbors_title: str,
                subset_title: str,
                data_title: str,
                color_title: str,
                show: bool = False,
                write: bool = False):
        fig = px.scatter(
            umap_proj_2d, x=0, y=1,
            title=f'UMAP of {data_title} colored by {color_title} '
                  f'(neighbors = {n_neighbors_title}, subset = {subset_title})',
            color=color,
            labels={'color': color_title},
            hover_data=hover_data,
            width=1800, height=1000
        )
        if show:
            fig.show()
        if write:
            fig.write_image(
                f'../plots/umap2d_{data_title.lower()}_t{subset_title}_n{n_neighbors_title}_{color_title.lower()}.png')

    @staticmethod
    def umap_3d(umap_proj_3d: np.array, color: pd.Series,
                hover_data: Dict[str, any],
                n_neighbors_title: str,
                subset_title: str,
                data_title: str,
                color_title: str,
                show: bool = False,
                write: bool = False):
        fig = px.scatter_3d(
            umap_proj_3d, x=0, y=1, z=2,
            title=f'UMAP of {data_title} colored by {color_title} '
                  f'(neighbors = {n_neighbors_title}, subset = {subset_title})',
            color=color,
            labels={'color': color_title},
            hover_data=hover_data,
            width=1800, height=1000
        )
        if show:
            fig.show()
        if write:
            fig.write_image(
                f'../plots/umap3d_{data_title.lower()}_t{subset_title}_n{n_neighbors_title}_{color_title.lower()}.png')

    @staticmethod
    def show_pdb(pdb_file: str, color: str, bind_annot_names: list = None, show_sidechains: bool = False,
                 show_mainchains: bool = False) -> py3Dmol.view:
        with open(pdb_file, 'r') as f:
            system = "".join([x for x in f])
        view = py3Dmol.view(js='https://3dmol.org/build/3Dmol.js', )
        view.addModel(system)

        possible_colors = {'lDDT', 'rainbow', 'ligand'}
        assert color in possible_colors, f'Invalid color given. Choose one of {str(possible_colors)}'

        if color == "lDDT":
            view.setStyle({'cartoon': {'colorscheme': {'prop': 'b', 'gradient': 'roygb', 'min': 50, 'max': 90}}})
        elif color == "rainbow":
            view.setStyle({'cartoon': {'color': 'spectrum'}})
        elif color == "ligand":
            assert bind_annot_names is not None, 'bind_annot_ids should not be None if color="bind_annot"'

            i = 0
            for line in system.split("\n"):
                split = line.split()
                if len(split) == 0 or split[0] != "ATOM":
                    continue
                bind_annot = bind_annot_names[int(split[5]) - 1]
                if bind_annot not in BIND_ANNOT_COLORS.keys():
                    color = 'purple'
                else:
                    color = BIND_ANNOT_COLORS[bind_annot]
                view.setStyle({'model': -1, 'serial': i + 1}, {"cartoon": {'color': color}})
                i += 1

        if show_sidechains:
            BB = ['C', 'O', 'N']
            view.addStyle({'and': [{'resn': ["GLY", "PRO"], 'invert': True}, {'atom': BB, 'invert': True}]},
                          {'stick': {'colorscheme': f"WhiteCarbon", 'radius': 0.3}})
            view.addStyle({'and': [{'resn': "GLY"}, {'atom': 'CA'}]},
                          {'sphere': {'colorscheme': f"WhiteCarbon", 'radius': 0.3}})
            view.addStyle({'and': [{'resn': "PRO"}, {'atom': ['C', 'O'], 'invert': True}]},
                          {'stick': {'colorscheme': f"WhiteCarbon", 'radius': 0.3}})
        if show_mainchains:
            BB = ['C', 'O', 'N', 'CA']
            view.addStyle({'atom': BB}, {'stick': {'colorscheme': f"WhiteCarbon", 'radius': 0.3}})

        view.zoomTo()
        view.show()
        return view

    @staticmethod
    def plot_plddt_legend(dpi=100):
        thresh = ['plDDT:', 'Very low (<50)', 'Low (60)', 'OK (70)', 'Confident (80)', 'Very high (>90)']
        plt.figure(figsize=(1, 0.1), dpi=dpi)
        ########################################
        for c in ["#FFFFFF", "#FF0000", "#FFFF00", "#00FF00", "#00FFFF", "#0000FF"]:
            plt.bar(0, 0, color=c)
        plt.legend(thresh, frameon=False,
                   loc='center', ncol=6,
                   handletextpad=1,
                   columnspacing=1,
                   markerscale=0.5, )
        plt.axis(False)
        return plt

    @staticmethod
    def plot_ligand_legend(dpi=100):
        ligands = []
        colors = []
        for ligand, color in BIND_ANNOT_COLORS.items():
            ligands.append(ligand)
            colors.append(color)
        plt.figure(figsize=(1, 0.1), dpi=dpi)
        ########################################
        for c in ["#FFFFFF"] + colors:
            plt.bar(0, 0, color=c)
        plt.legend(['ligand:'] + ligands, frameon=False,
                   loc='center', ncol=len(ligands) + 1,
                   handletextpad=1,
                   columnspacing=1,
                   markerscale=0.5, )
        plt.axis(False)
        return plt

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_label: str, ax: plt.axes, fontsize=14):
        cf = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1])

        try:
            group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
            group_counts = ['{0: 0.0f}'.format(value) for value in cf.flatten()]
            group_percentages = ['{0: 0.2%}'.format(value) for value in cf.flatten() / np.sum(cf)]
            labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
            labels = np.asarray(labels).reshape(2, 2)
            heatmap = sns.heatmap(cf, annot=labels, fmt="", cbar=False, square=True, cmap='Blues', ax=ax)
        except ValueError:
            raise ValueError("Confusion matrix values must be integers.")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
        heatmap.set_title('Confusion matrix - ' + class_label)

    @staticmethod
    def plot_roc(y_true, y_score, class_label: str, ax: plt.axes):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        lw = 2
        ax.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=lw,
            label="ROC curve (area = %0.2f)" % roc_auc,
        )
        ax.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC curve - " + class_label)
        ax.legend(loc="lower right")

    @staticmethod
    def plot_ri_hist(x, ax: plt.axes, class_label: str) -> plt.figure:
        ax.hist(x)
        ax.set_title(class_label)
        ax.set_xlabel("RI")
        ax.set_ylabel("Counts")

    @staticmethod
    def plot_plddt_ecdf(plddt_tensor, ax=plt.axes):
        ax.plot(np.sort(plddt_tensor), np.linspace(0, 1, len(plddt_tensor), endpoint=False))
        ax.set_title('pLDDT Distribution')
        ax.set_xlabel("pLDDT")
        ax.set_ylabel("Probability")

    @staticmethod
    def plot_protein_length_hist(protein_length, ax=plt.axes):
        ax.hist(protein_length)
        ax.set_title(f'Protein length (total proteins={str(len(protein_length))})')
        ax.set_xlabel("Protein length")
        ax.set_ylabel("Count")

    @staticmethod
    def plot_performance_per_cutoff(metrics_per_cutoff: Dict[float, dict], ax: plt.axes, class_label: str):
        rec = []
        prec = []
        covonebind = []
        cutoffs = []
        for cutoff, metrics in metrics_per_cutoff.items():
            cutoffs.append(cutoff)
            rec.append(metrics['rec_' + class_label])
            prec.append(metrics['prec_' + class_label])
            covonebind.append(metrics['covonebind_' + class_label])
        ax.plot(cutoffs, rec, color='green', label='Recall')
        ax.plot(cutoffs, prec, color='blue', label='Precission')
        ax.plot(cutoffs, covonebind, color='orange', label="CovOneBind")
        ax.set_title(class_label)
        ax.set_xlabel("Probability")
        ax.set_ylabel("Performance")
        ax.legend()
