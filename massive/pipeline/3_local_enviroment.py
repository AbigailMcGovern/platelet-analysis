import pandas as pd
import argparse
from plateletanalysis.variables.neighbours import add_neighbour_lists, local_density, local_calcium

p = argparse.ArgumentParser()
p.add_argument('-f', '--file', help='parquet file platelet info')
p.add_argument('-s', '--save', help='parquet file into which to save output')
args = p.parse_args()

p = args.file
sp = args.save
df = pd.read_parquet(p)

df = add_neighbour_lists(df)
df = local_density(df)
df = local_calcium(df)

df.to_parquet(sp)

