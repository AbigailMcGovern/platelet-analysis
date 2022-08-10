from plateletanalysis.variables.transform import revert_to_pixel_coords
from plateletanalysis.variables.measure import assign_fibrin_distance
import pandas as pd
import zarr
import os
from pathlib import Path
import argparse


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


p = argparse.ArgumentParser()
p.add_argument('-d', '--data', help='platelet dataframe')
p.add_argument('-i', '--images', help='image data directory')
p.add_argument('-m', '--metadata', help='metadata directory')
p.add_argument('-s', '--save', help='parquet file into which to save output')
args = p.parse_args()

df = pd.read_parquet(p)


# Paths to data and directories
md_d = args.metadata
dt_d = args.images
data_path = args.data

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
sp = args.save
df.to_parquet(sp)