from plateletanalysis.variables.neighbours import add_neighbour_lists, local_density, local_contraction
from plateletanalysis.variables.measure import path_disp_n_tortuosity
import os
import pandas as pd
import argparse


p = argparse.ArgumentParser()
p.add_argument('-f', '--file', help='parquet file platelet info')
p.add_argument('-s', '--save', help='parquet file into which to save output')
args = p.parse_args()



p = args.file
sp = args.save
df = pd.read_parquet(p)


df = add_neighbour_lists(df)
df.to_parquet(sp)

df = local_density(df)
df.to_parquet(sp)


df['nrterm'] = df['nrtracks'] - df['tracknr']
df['terminating'] = df['nrtracks'] == df['tracknr']
df.to_parquet(sp)

df = path_disp_n_tortuosity(df)
df.to_parquet(sp)
