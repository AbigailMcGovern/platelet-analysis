import pandas as pd
import argparse


p = argparse.ArgumentParser()
p.add_argument('-f', '--file', help='parquet file platelet info')
p.add_argument('-s', '--save', help='parquet file into which to save output')
args = p.parse_args()

p = args.file
sp = args.save
df = pd.read_parquet(p)

df = df.sort_values('frame')

for key, grp in df.groupby(['path', 'particle']):
    idx = grp.index.values
    df.loc[idx, 'x_corr'] = grp['x_s'].rolling(5, win_type='triang',min_periods=1).mean()
    df.loc[idx, 'y_corr'] = grp['ys'].rolling(5, win_type='triang',min_periods=1).mean()
    df.loc[idx, 'z_corr'] = grp['zs'].rolling(5, win_type='triang',min_periods=1).mean()

df.to_parquet(sp)
