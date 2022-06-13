import numpy as np
import pandas as pd
import plotly.express as px
from typing import Dict


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
