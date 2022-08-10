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
    new_x = grp['x_s'].rolling(5, win_type='triang',min_periods=1).mean()
    df.loc[idx, 'x_s_orig'] = grp['x_s'].values
    df.loc[idx, 'x_s'] = new_x
    new_y = grp['ys'].rolling(5, win_type='triang',min_periods=1).mean()
    df.loc[idx, 'ys_orig'] = grp['ys'].values
    df.loc[idx, 'ys'] = new_y
    new_z = grp['zs'].rolling(5, win_type='triang',min_periods=1).mean()
    df.loc[idx, 'zs_orig'] = grp['zs'].values
    df.loc[idx, 'zs'] = new_z

df.to_parquet(sp)
