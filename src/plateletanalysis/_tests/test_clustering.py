from plateletanalysis.analysis.clustering import cluster_spherical_coordinates, plot_clusters
import pandas as pd
import os
from pathlib import Path

dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
pa_dir = dir_path.parents[0] # plateletanalysis directory

def read_data():
    data_path = os.path.join(pa_dir, 'data', 'test_data', 'test_data.parquet') # path to test data 
    # test data is frames 0-1 for 191016_IVMTR12_Inj1_ctrl_exp3
    df = pd.read_parquet(data_path)
    return df


def test_clustering_spherical_coords():
    df = read_data()
    df = cluster_spherical_coordinates(df)
    return df


def test_plot_clusters():
    df = test_clustering_spherical_coords()
    df = plot_clusters(df, 'dbscan_1')
    df = plot_clusters(df, 'dbscan_3', umap_cols=['umap_0_scoords', 'umap_1_scoords'])
    
    ...


if __name__ == '__main__':
    #df = test_clustering_spherical_coords()
    test_plot_clusters()
    #eps_list = []
    #for eps in eps_list:
     #   u_labs = pd.unique(df[f'dbscan_{eps}'])
