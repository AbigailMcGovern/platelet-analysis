from ripser import ripser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import seaborn as sns
from pathlib import Path
from scipy import stats
from plateletanalysis.variables.measure import quantile_normalise_variables, quantile_normalise_variables_frame
#from plateletanalysis.variables.position import scale_free_positional_categories, count_variables, transitions
from scipy.signal import find_peaks
from plateletanalysis.variables.neighbours import add_neighbour_lists, local_density
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from plateletanalysis.variables.basic import get_treatment_name



# -----------------------------------------------------------------------------
# ------------------------------
# Major data aquisition function 
# ------------------------------
# -----------------------------------------------------------------------------

def donutness_data(
        df : pd.DataFrame, 
        sample_col='path', 
        time_col='frame',
        sample_n=200,
        n_samples=100,
        centile=50,
        filter_col='nb_density_15_pcntf', 
        y_col='ys', 
        x_col='x_s', 
        #get_accessory_data=False, 
        units='%'
    ):
    # only the upper centiles of the data (e.g., density, fibrin, p selectin)
    df = scale_x_and_y(df, x_col, y_col)
    if filter_col is not None:
        if isinstance(centile, int) or isinstance(centile, float):
            df = df[df[filter_col] > centile]
        elif isinstance(centile, tuple):
            df = df[(df[filter_col] > centile[0]) & (df[filter_col] > centile[1])]
    y_col_s = y_col + '_scaled'
    x_col_s = x_col + '_scaled'
    data_cols = [sample_col, x_col_s, y_col_s]
    if time_col is not None:
        data_cols.append(time_col)
    df = df[data_cols]
    samples = pd.unique(df[sample_col])
    ph_data = initialise_PH_data_dict(sample_col, time_col)
    samples = pd.unique(df[sample_col])
    #tx_name = get_treatment_name(data['path'].values[0])
    #desc=f'Getting max barcode data for treatment = {tx_name}'
    frames = frames_list(df, time_col)
    n_its = len(samples) * n_samples * len(frames)
    with tqdm(total=n_its) as progress:
        for sample, data in df.groupby(sample_col):
            for t in frames:
                df_t = frame_df(data, time_col, t)
                for bs_id in range(n_samples):
                    if len(df_t) > 0:
                        df_t_s = df_t.sample(n=sample_n, replace=True)
                        ph_data[time_col].append(t)
                        sample_persistent_homology_analysis(df_t_s, x_col_s, y_col_s, 
                                                        ph_data, sample_col, 
                                                        time_col, sample, bs_id)
                    else:
                        ph_data[time_col].append(t)
                        assign_nan(ph_data, bs_id, time_col, sample_col, sample)
                        
                    progress.update(1)
    ph_data = pd.DataFrame(ph_data)
    #if get_accessory_data:
     #   _, donut_info = find_max_donut_time(out)
      #  out = accessory_platelet_data(out, df, donut_info)
    ph_data[ph_data['outlierness_mean'] == 0]['donutness'] = 0
    return ph_data



def initialise_PH_data_dict(sample_col, time_col):
    ph_data = {
        'bootstrap_id' : [],
        'birth_1' : [],
        'birth_2' : [],
        'birth_mean' : [],
        'birth_std' : [],
        'death_1' : [], 
        'death_2' : [], 
        'death_mean' : [],
        'death_std' : [],
        'persistence_1' : [],
        'persistence_2' : [], 
        'persistence_mean' : [],
        'persistence_std' : [],
        'outlierness_1' : [], 
        'outlierness_2' : [],
        'outlierness_mean' : [],
        'outlierness_std' : [],
        'donutness' : []
    }
    ph_data[sample_col] = []
    if time_col is not None:
        ph_data[time_col] = []
    return ph_data

def assign_nan(ph_data, bs_id, time_col, sample_col, sample):
    length = len(ph_data[time_col])
    vars = ['birth_1', 'birth_2', 'birth_mean', 
            'birth_std', 'death_1', 'death_2', 'death_mean', 'death_std',
            'persistence_1', 'persistence_2', 'persistence_mean', 
            'persistence_std', 'outlierness_1', 'outlierness_2', 
            'outlierness_mean', 'outlierness_std', 'donutness']
    ph_data['bootstrap_id'].append(bs_id)
    ph_data[sample_col].append(sample)
    for var in vars:
        ph_data[var].append(np.NaN)
    assert len(ph_data[sample_col]) == length
    assert len(ph_data['bootstrap_id']) == length
    for v in vars:
        assert len(ph_data[v]) == length


def frames_list(data, time_col):
    if time_col is not None:
        frames = list(pd.unique(data[time_col]))
        frames.sort()
    else:
        frames = [0, ]
    return frames


def frame_df(data, time_col, t):
    if time_col is not None:
        data_t = data[data[time_col] == t]
    else:
        data_t = data
    return data_t

def sample_persistent_homology_analysis(
        data, 
        x_col, 
        y_col, 
        ph_data, 
        sample_col, 
        time_col, 
        sample, 
        bootstrap_id
    ):
    X = data[[x_col, y_col]].values
    ph_data[sample_col].append(sample)
    ph_data['bootstrap_id'].append(bootstrap_id)
    if len(X) > 0:
        dgms = ripser(X)['dgms']
        h1 = dgms[1]
        #print(h1)
        if len(h1) > 1:
            births = h1[:, 0]
            deaths = h1[:, 1]
            persistence = deaths - births
            idxs = np.argsort(persistence)
            # births
            ph_data['birth_1'].append( births[idxs[-1]])
            ph_data['birth_2'].append(births[idxs[-2]])
            ph_data['birth_mean'].append(np.mean(births))
            ph_data['birth_std'].append(np.std(births))
            # deaths
            ph_data['death_1'].append(deaths[idxs[-1]])
            ph_data['death_2'].append(deaths[idxs[-2]])
            ph_data['death_mean'].append(np.mean(deaths))
            ph_data['death_std'].append(np.std(deaths))
            # persistence
            ph_data['persistence_1'].append(persistence[idxs[-1]])
            ph_data['persistence_2'].append(persistence[idxs[-2]])
            persistence_mean = np.mean(persistence)
            ph_data['persistence_mean'].append(persistence_mean)
            persistence_std = np.std(persistence)
            ph_data['persistence_std'].append(persistence_std)
            # outlierness
            outlierness = (persistence - persistence_mean) / persistence_std
            outlierness_1 = outlierness[idxs[-1]]
            ph_data['outlierness_1'].append(outlierness_1)
            outlierness_2 = outlierness[idxs[-2]]
            ph_data['outlierness_2'].append(outlierness_2)
            ph_data['outlierness_mean'].append(np.mean(outlierness))
            ph_data['outlierness_std'].append(np.std(outlierness))
            # donutness
            ph_data['donutness'].append(outlierness_1 - outlierness_2)
        else:
            append_NaN(ph_data, sample_col, time_col)
    else:
        append_NaN(ph_data, sample_col, time_col)

    return ph_data



def append_NaN(ph_data, sample_col, time_col):
    for k in ph_data.keys():
        if k != sample_col and k != time_col and k != 'bootstrap_id':
            ph_data[k].append(np.NaN)




# ---------------
# Additional Data
# ---------------

#TODO: Please change this function to add mean data to existing dataframe
# -i.e., move this to another module and make usable for any summary table:
#           - with path + frame
#           - for any categorical/condition 
#           (e.g., exp @ frame 100, only surface platelets) 

def accessory_platelet_data(
        out, 
        df, 
        donut_info,
        positional_cols=('surface_or_core', 'surface_or_core', 'anterior_surface', 'tail', 'donut'),
        conditions=('surface', 'core', True, True, True)
    ):
    df = df[df['nrtracks'] > 2]
    if 'treatment' in df.columns.values:
        df = df.drop('treatment', axis=1)
    df['treatment'] = df['path'].apply(get_treatment_name)
    #df = scale_free_positional_categories(df, donut_df=donut_info)
    uframes = pd.unique(out['frame'])
    upaths = pd.unique(out['path'])
    its = len(upaths) * len(uframes)
    with tqdm(desc='Adding averages and count variables', total=its) as progress:
        for k, grp in df.groupby(['path', 'frame']):
            p = k[0]
            f = k[1]
            odf = out[(out['path'] == p) & (out['frame'] == f)]
            idx = odf.index.values
            out = _add_averages(out, grp, idx)
            ndf = df[(df['path'] == p) & (df['frame'] == f + 1)]
            for i, col in enumerate(positional_cols):
                cond = conditions[i]
                if isinstance(cond, str):
                    pre = cond + ' '
                else:
                    pre = col + ' '
                out = _add_averages(out, grp, idx, pre, col, cond)
                #out = count_variables(out, grp, ndf, idx, col, cond, pre)
            progress.update(1)
    return out


def _add_averages(
        out, 
        df, 
        idx, 
        prefix='',
        column=None,
        condition=None,
        variables=('dv', 'dvz', 'dvy', 'dvx', 'ca_corr', 'dist_c', 'nb_density_15', 'ca_corr_pcnt', 'cont', 'elong'), 
        variable_names=('dv (um/s)', 'dvz (um/s)', 'dvy (um/s)', 'dvx (um/s)', 'corrected calcium', 
                        'centre distance', 'density (platelets/um^2)', 'corrected calcium (%)', 'contraction (um/s)', 'elongation')
    ):
    if column is not None:
        df = df[df[column] == condition]
    for i, v in enumerate(variables):
        n = prefix + variable_names[i]
        out.loc[idx, n] = df[v].mean()
    return out


# -------
# Helpers
# -------

def get_count(df, thresh_col='nd15_percentile', threshold=25):
    sml_df = df[df[thresh_col] > threshold]
    count = len(sml_df)
    return count



def scale_x_and_y(df, x_col='x_s', y_col='ys'):
    if 'nrtracks' in df.columns.values:
        df = df[df['nrtracks'] > 10]
    #if 'nb_density_15_pcntf' not in df.columns.values:
     #   df = quantile_normalise_variables_frame(df, ('nb_density_15', ))
    df = scale_data(df, x_col)
    df = scale_data(df, y_col)
    return df



def scale_data(df, col, groupby_list=['path', 'frame']):
    for k, g in df.groupby(groupby_list):
        scaler = StandardScaler()
        data = g[col].values
        data = np.expand_dims(data, 1)
        scaler.fit(data)
        new = scaler.transform(data)
        new = np.squeeze(new)
        idx = g.index.values
        df.loc[idx, f'{col}_scaled'] = new
    return df


def summary_donutness(ddf):
    ddf = ddf[ddf['outlierness_mean'] != 0] # error when there are only two loops 
    res = defaultdict(list)
    for k, grp in ddf.groupby(['path', 'time (s)']):
        res['path'].append(k[0])
        res['time (s)'].append(k[1])
        res['donutness'].append(grp['donutness'].mean())
    res = pd.DataFrame(res)
    return res


def average_donutness_vs_count_plot(ddf, cdf):
    result = defaultdict(list)
    for k, grp in ddf.groupby('path'):
        result['path'].append(k)
        result['donutness'].append(grp['donutness'].mean())
        result['count'].append(cdf[cdf['path'] == k]['platelet count'].mean())
    result = pd.DataFrame(result)
    sns.scatterplot(data=result, x='donutness', y='count', hue='path')
    plt.show()
