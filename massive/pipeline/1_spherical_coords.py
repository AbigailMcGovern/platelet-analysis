import pandas as pd
import argparse
from plateletanalysis.variables.transform import spherical_coordinates


p = argparse.ArgumentParser()
p.add_argument('-f', '--file', help='parquet file platelet info')
p.add_argument('-s', '--save', help='parquet file into which to save output')
args = p.parse_args()

p = args.file
sp = args.save
df = pd.read_parquet(p)


# the smoothing causes an artefact at the end of each track
df['nrterm'] = df['nrtracks'] - df['tracknr']
# eliminate the last 5 tracks of each 
df = df[df['nrterm'] > 4]

df = spherical_coordinates(df)

df.to_parquet(sp)