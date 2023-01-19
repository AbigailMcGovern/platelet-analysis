import pandas as pd
import argparse


def cumulative(df, col):
    df = df.sort_values('frame')
    for key, grp in df.groupby(['path', 'particle']):
        idx = grp.index.values
        cumsum = grp[col].cumsum()
        n = col + '_csum'
        df.loc[idx, n] = cumsum
    return df


p = argparse.ArgumentParser()
p.add_argument('-f', '--file', help='parquet file platelet info')
p.add_argument('-s', '--save', help='parquet file into which to save output')
args = p.parse_args()

p = args.file
sp = args.save
df = pd.read_parquet(p)

df = cumulative(df, 'ca_corr')
df = cumulative(df, 'dv')
df = cumulative(df, 'ca_corr')

df.to_parquet(sp)
