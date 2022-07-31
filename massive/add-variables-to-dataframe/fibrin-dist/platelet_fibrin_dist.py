from plateletanalysis.variables.transform import revert_to_pixel_coords
from plateletanalysis.variables.measure import assign_fibrin_distance
import pandas as pd
import zarr
import os
from pathlib import Path


def get_fibrin_dts(d, ls=None):
    if ls is None:
        files = [f for f in os.listdir(d) if f.endswith('_fibrin-dist.zarr')]
    else:
        files = ls
    segmentations = {}
    for f in files:
        sp = os.path.join(d, f)
        seg = zarr.open(sp)
        n = Path(sp).stem
        n = n[:n.find('_fibrin-dist')]
        segmentations[n] = seg
    return segmentations


# Paths to data and directories
md_d = '/fs02/rl54/results/210920_141056_seg-track_platelet-tracks'
dt_d = '/fs02/rl54/data/Inhibitor cohort 2020/Saline'
data_path = '/fs02/rl54/dataframes/211206_saline_df_spherical-coords_density_pcnt.parquet'

# get the metadata to revert to pixel coordinates
md_files = [os.path.join(md_d, f) for f in os.listdir(md_d) if f.endswith('md.csv') and f.find('_saline_') != -1]
md_dfs = [pd.read_csv(p) for p in md_files]

# add columns into the data frame with each platelet's original pixel coordinates
df = pd.read_parquet(data_path)
df = revert_to_pixel_coords(df, md_dfs)

# obtain a dict with the paths and distance transforms for each timeseries
dt_dict = get_fibrin_dts(dt_d)

# assign fibrin distance
df = assign_fibrin_distance(df, dt_dict)

# save to a new file
sp = '/fs02/rl54/dataframes/211206_saline_df_220610.parquet'
df.to_parquet(sp)