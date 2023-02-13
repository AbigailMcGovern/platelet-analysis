import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


# ----------
# data funcs
# ----------

def first_2_frames(df):
    sdf = df[df['tracknr'] < 3]
    return sdf


def first_5_frames(df):
    sdf = df[df['tracknr'] < 6]
    return sdf



# -------------
# Main function
# -------------

def boxplots_data(
        df, 
        save_path, 
        variables, 
        funcs={'first 2 frames' : first_2_frames, 'first 5 frames' : first_5_frames}, 
        track_lim=1, 
        phases={'growth' : (0, 260), 'consolidation' : (260, 600)},
        groupby=['phase', 'path', 'shedding'], # can add region
        ):
    df = df[df['nrtracks'] > track_lim]
    df['phase'] = [None, ] * len(df)
    df = add_phase(df, phases)
    out = {c : [] for c in groupby}
    var_funcs = {}
    for v in variables:
        for f in funcs.keys():
            n = f"{v} : {f}"
            var_funcs[n] = (v, funcs[f])
            out[n] = []
    nits = 0
    for k, g in df.groupby(groupby):
        nits += 0
    with tqdm(total=nits) as progress:
        for k, g in df.groupby(groupby):
            for n in var_funcs.keys():
                for i, col in enumerate(groupby):
                    out[col].append(k[i])
                var, func = var_funcs[n]
                sdf = func(g)
                for v in variables:
                    out[var].append(sdf[v].mean())
            progress.update(1)
    out = pd.DataFrame(out)
    out.to_csv(save_path)
    return out



# -------------
# Add variables
# -------------

def add_phase(df, phases={'growth' : (0, 260), 'consolidation' : (260, 600)}):
    df['phase'] = [None, ] * len(df)
    for phase in phases:
        sdf = df[(df['time (s)'] > phases[phase][0]) & (df['time (s)'] > phases[phase][1])]
        idxs = sdf.index.values
        df.loc[idxs, 'phase'] = phase
    return df



def add_time_seconds(df, frame_col='frame'):
    df['time (s)'] = df[frame_col] / 0.321764322705706
    return df


def add_sliding_variable(df):
    df['sliding (ums^-1)'] = [None, ] * len(df)
    # not moving in direction of blood flow
    sdf = df[df['dvy'] >= 0]
    idxs = sdf.index.values
    df.loc[idxs, 'sliding (ums^-1)'] = 0
    # moving in the direction of blood floe
    sdf = df[df['dvy'] < 0]
    idxs = sdf.index.values
    new = np.abs(sdf['dvy'].values)
    df.loc[idxs, 'sliding (ums^-1)'] = new
    return df


def tracking_time_var(df):
    df['tracking time (s)'] = df['tracknr'] / 0.321764322705706
    return df


def time_tracked_var(df):
    df['total time tracked (s)'] = df['nrtracks'] / 0.321764322705706
    return df


def add_terminating(df):
    df['terminating'] = [False, ] * len(df)
    for k, g in df.groupby(['path', ]):
        t_max = g['frame'].max()
        sdf = g[g['frame'] != t_max]
        term = sdf['nrtracks'] == sdf['tracknr']
        idxs = sdf.index.values
        df.loc[idxs, 'terminating'] = term
    return df


def add_normalised_ca_pcnt(df):
    for k, g in df.groupby(['path', 'frame']):
        ca_max = g['ca_corr'].max()
        ca_norm = g['ca_corr'] / ca_max * 100
        idxs = g.index.values
        df.loc[idxs, 'Ca2+ pcnt max'] = ca_norm
    return df


def add_shedding(df):
    df['shedding'] = [False, ] * len(df)
    nits = 0
    for k, g in df.groupby(['path', 'particle']):
        nits += 1
    with tqdm(total=nits) as progress:
        for k, g in df.groupby(['path', 'particle']):
            #shed = df['terminating'].sum() # was going to take 11 hours
            if True in df['terminating'].values:
                idxs = g.index.values
                df.loc[idxs, 'shedding'] = True
            progress.update(1)
    return df



# --------
# Plotting
# --------


def plot_boxplot(
        data, 
        variables,
        treatment_col='treatment', 
        hue='shedding'
        ):
    fig, axs = plt.subplots(len(variables), 1)
    for i, ax in enumerate(axs.ravel()):
        sns.boxplot(data=data, x=treatment_col, y=variables[i], hue=hue, ax=ax)
    plt.show()




if __name__ == '__main__':
    from plateletanalysis.variables.basic import get_treatment_name, time_minutes

    # --------
    # Get data
    # --------
    # takes about 1-2 min to read in data 
    d = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes'
    file_names = ('211206_mips_df.parquet', '211206_veh-mips_df.parquet', 
                  '211206_cang_df.parquet', '211206_saline_df_220827_amp0.parquet', 
                  '211206_sq_df.parquet', '211206_veh-sq_df.parquet')
    file_paths = [os.path.join(d, n) for n in file_names]
    #dfs = [pd.read_parquet(p) for p in file_paths]
    data = []
    for p in file_paths:
        df = pd.read_parquet(p)
        data.append(df)
    df = pd.concat(data).reset_index(drop=True)
    del data


    # -------------
    # Add variables
    # -------------
    
    df['treatment'] = df['path'].apply(get_treatment_name)
    df = df[df['treatment'] != 'DMSO (salgav)']
    print('Adding variables...')
    with tqdm(total=8) as progress: # ~ 1 min for first 7, add_shedding() takes 
        df = add_time_seconds(df)
        progress.update(1)
        df = add_terminating(df)
        progress.update(1)
        df = add_sliding_variable(df)
        progress.update(1)
        df = time_minutes(df)
        progress.update(1)
        df = time_tracked_var(df)
        progress.update(1)
        df = tracking_time_var(df)
        progress.update(1)
        df = add_normalised_ca_pcnt(df)
        progress.update(1)
        df = add_shedding(df)
        progress.update(1)
    df.to_parquet(os.path.join(d, 'MIPS_paper_with_shedding.parquet'))

    # ----------------------
    # Obtain data for graphs
    # ----------------------
    save_path = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/shedding_vs_non/230212_sheddingvsnon_growth_consol_.csv'
    variables = ('ca_corr', 'Ca2+ pcnt max', 'sliding (ums^-1)', 'total time tracked (s)', 'nb_density_15')
    out = boxplots_data(df, save_path, variables) 
    #out = pd.read_csv(save_path)
    plot_boxplot(out, ('Ca2+ pcnt max', 'sliding (ums^-1)', 'nb_density_15'))