import numpy as np
from plots import Plots
import pandas as pd
from umap import UMAP


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
                              'ligand': long_df_head.bind_annot_name,
                              'position': long_df_head.position,
                              'residue': long_df_head.residue},
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
                              'ligand': long_df_head.bind_annot_name,
                              'position': long_df_head.position,
                              'residue': long_df_head.residue},
                  n_neighbors_title=str(n_neighbors),
                  subset_title=str(n),
                  write=write,
                  show=show)

    # plot by ligand
    print('UMAP Ligand')
    Plots.umap_2d(umap_proj_2d=umap_proj_2d,
                  data_title=data_title,
                  color_title='Ligand',
                  color=long_df_head.bind_annot_name,
                  hover_data={'Amino acid': long_df_head.residue,
                              'Protein': long_df_head.protein_id,
                              'position': long_df_head.position,
                              'residue': long_df_head.residue},
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
                              'ligand': long_df_head.bind_annot_name},
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
                              'ligand': long_df_head.bind_annot_name},
                  n_neighbors_title=str(n_neighbors),
                  subset_title=str(n),
                  write=write,
                  show=show)

    # plot by ligand
    print('UMAP Ligand')
    Plots.umap_3d(umap_proj_3d=umap_proj_3d,
                  data_title=data_title,
                  color_title='Ligand',
                  color=long_df_head.bind_annot_name,
                  hover_data={'Amino acid': long_df_head.residue,
                              'Protein': long_df_head.protein_id},
                  n_neighbors_title=str(n_neighbors),
                  subset_title=str(n),
                  write=write,
                  show=show)


def umap_plots(long_embeddings: np.array, long_df: pd.DataFrame, long_distograms: np.array, n_neighbors: int, n: int,
               write: bool, show: bool, plot_3D=True):
    # Embeddings
    umap_plots_2d(X=long_embeddings, long_df_head=long_df, data_title='Embeddings',
                  n_neighbors=n_neighbors, n=n, show=show, write=write)
    if plot_3D:
        umap_plots_3d(X=long_embeddings, long_df_head=long_df, data_title='Embeddings',
                      n_neighbors=n_neighbors, n=n, show=show, write=write)
    # Distograms
    umap_plots_2d(X=long_distograms, long_df_head=long_df, data_title='Distograms',
                  n_neighbors=n_neighbors, n=n, show=show, write=write)
    if plot_3D:
        umap_plots_3d(X=long_distograms, long_df_head=long_df, data_title='Distograms',
                      n_neighbors=n_neighbors, n=n, show=show, write=write)
