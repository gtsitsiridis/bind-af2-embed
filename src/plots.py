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

def umap_plots_2d(X: np.array, long_df_head: pd.DataFrame, data_title: str, n_neighbors: int, n: int, show: bool,
                  write: bool):
    print(f'UMAP 2D: {data_title}-subset{str(n)}-neighbors{n_neighbors}')
    umap_2d = UMAP(n_neighbors=n_neighbors, n_components=2, init='random', random_state=42, verbose=True)
    umap_proj_2d = umap_2d.fit_transform(X)
    np.save(f'../data/output/umap/umap2d_{data_title}_subset{str(n)}_neighbors{str(n_neighbors)}', umap_proj_2d)

    ## plot by amino acid
    print('UMAP AA')
    Plots.umap_2d(umap_proj_2d=umap_proj_2d,
                  data_title=data_title,
                  color_title='Amino acid',
                  color=long_df_head.residue,
                  hover_data={'protein ID': long_df_head.protein_id,
                              'ligand': long_df_head.label_name},
                  n_neighbors_title=str(n_neighbors),
                  subset_title=str(n),
                  write=write,
                  show=show)

    # plot by protein
    print('UMAP Protein')
    Plots.umap_2d(umap_proj_2d=umap_proj_2d,
                  data_title=data_title,
                  color_title='Protein',
                  color=long_df_head.protein_id,
                  hover_data={'Amino acid': long_df_head.residue,
                              'ligand': long_df_head.label_name},
                  n_neighbors_title=str(n_neighbors),
                  subset_title=str(n),
                  write=write,
                  show=show)

    # plot by ligand
    print('UMAP Ligand')
    Plots.umap_2d(umap_proj_2d=umap_proj_2d,
                  data_title=data_title,
                  color_title='Ligand',
                  color=long_df_head.label_name,
                  hover_data={'Amino acid': long_df_head.residue,
                              'Protein': long_df_head.protein_id},
                  n_neighbors_title=str(n_neighbors),
                  subset_title=str(n),
                  write=write,
                  show=show)


def umap_plots_3d(X: np.array, long_df_head: pd.DataFrame, data_title: str, n_neighbors: int, n: int, show: bool,
                  write: bool):
    print(f'UMAP 3D: {data_title}-subset{str(n)}-neighbors{n_neighbors}')
    umap_3d = UMAP(n_neighbors=n_neighbors, n_components=3, init='random', random_state=42, verbose=True)
    umap_proj_3d = umap_3d.fit_transform(X)
    np.save(f'../data/output/umap/umap3d_{data_title}_subset{str(n)}_neighbors{str(n_neighbors)}', umap_proj_3d)

    ## plot by amino acid
    print('UMAP AA')
    Plots.umap_3d(umap_proj_3d=umap_proj_3d,
                  data_title=data_title,
                  color_title='Amino acid',
                  color=long_df_head.residue,
                  hover_data={'protein ID': long_df_head.protein_id,
                              'ligand': long_df_head.label_name},
                  n_neighbors_title=str(n_neighbors),
                  subset_title=str(n),
                  write=write,
                  show=show)

    # plot by protein
    print('UMAP Protein')
    Plots.umap_3d(umap_proj_3d=umap_proj_3d,
                  data_title=data_title,
                  color_title='Protein',
                  color=long_df_head.protein_id,
                  hover_data={'Amino acid': long_df_head.residue,
                              'ligand': long_df_head.label_name},
                  n_neighbors_title=str(n_neighbors),
                  subset_title=str(n),
                  write=write,
                  show=show)

    # plot by ligand
    print('UMAP Ligand')
    Plots.umap_3d(umap_proj_3d=umap_proj_3d,
                  data_title=data_title,
                  color_title='Ligand',
                  color=long_df_head.label_name,
                  hover_data={'Amino acid': long_df_head.residue,
                              'Protein': long_df_head.protein_id},
                  n_neighbors_title=str(n_neighbors),
                  subset_title=str(n),
                  write=write,
                  show=show)
