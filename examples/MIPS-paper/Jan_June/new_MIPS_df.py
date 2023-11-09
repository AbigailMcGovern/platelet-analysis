import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from toolz import curry
from scipy import stats
from plateletanalysis.variables.measure import finite_difference_derivatives
from plateletanalysis.variables.basic import add_basic_variables, get_treatment_name
from plateletanalysis.variables.transform import adjust_coordinates
from plateletanalysis.variables.neighbours import local_density, add_neighbour_lists


dp = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS_raw/230209 MIPS segmentation output'

files = os.listdir(dp)
segtrack_paths = [os.path.join(dp, f) for f in files if f.endswith('tracks.csv')]


dfs = [pd.read_csv(sp) for sp in segtrack_paths]
df = pd.concat(dfs)
del dfs

md_path = os.path.join(dp, '230301_140207_segtrack_segmentation-metadata.csv')
md = pd.read_csv(md_path)


df['x_s'] = df['xs']
df['path'] = df['file']
df['frame'] = df['t']

df = adjust_coordinates(df, md)
df = finite_difference_derivatives(df)
#df = add_basic_variables(df)

sp = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/230301_MIPS_and_DMSO.parquet'

df.to_parquet(sp)

df = add_neighbour_lists(df)
df = local_density(df)

df['treatment'] = df['path'].apply(get_treatment_name)
df.to_parquet(sp)


dmso = df[df['treatment'] == 'DMSO (MIPS)']
mips = df[df['treatment'] == 'MIPS']

dmso.to_parquet('/Users/abigailmcgovern/Data/platelet-analysis/dataframes/230301_veh-MIPS.parquet')
mips.to_parquet('/Users/abigailmcgovern/Data/platelet-analysis/dataframes/230301_MIPS.parquet')


