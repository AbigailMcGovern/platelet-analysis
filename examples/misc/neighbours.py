from plateletanalysis.variables.neighbours import add_neighbour_lists, local_density
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Functions 
def add_n_neighbours(df):
    nbs = df['nb_particles_15'].apply(eval)
    nn = nbs.apply(len)
    df['n_neighbours'] = nn
    return df

def get_means(df, col='n_neighbours', groupby=['path', 'frame']):
    pass


def add_platelet_cum_sum(df, col='n_neighbours'):
    files = pd.unique(df['path'])
    for file in files:
        fdf = df[df['path'] == file]
        platelets = pd.unique(df['particle'])
        for p in platelets:
            pdf = fdf['particle' == p]


# Data path
d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
n = '211206_saline_df.parquet'
p = os.path.join(d, n)

# Read data
#df = pd.read_parquet(p)

# Add neighbour lists
#t = time()
#df = add_neighbour_lists(df)
#print(f'Added neighbour lists in {time() - t} seconds')

# Save progress
sp = os.path.join(d, '211206_saline_df_nb.parquet')
#df.to_parquet(sp)

# Load saved data

# add the number of neighbours
#df = add_n_neighbours(df)

# plot the number of neighbours versus frame
#frames = pd.unique(df['frame'])

#plt.plot(df['frame'].values, df['n_neighbours'].values)
sp = os.path.join(d, '211206_saline_df_nb_dens.parquet')
df = pd.read_parquet(sp)

df = local_density(df)
df.to_parquet(sp)

def add_mean_clot_lines(df, col='n_neighbours'):
    files = pd.unique(df['path'])
    for file in files:
        pass


df_gb_sem = df.groupby(['path', 'frame']).sem()
sem_frames = [i[1] for i in df_gb_sem.index.values]
sem_paths = [i[0] for i in df_gb_sem.index.values]
df_gb_sem['path'] = sem_paths
df_gb_sem['frame'] = sem_frames


