import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from plateletanalysis import add_basic_variables_to_files, add_region_category
from plateletanalysis.analysis.plots import pal1
import numpy as np
from scipy.signal import find_peaks, peak_widths, peak_prominences

def get_tracked_untracked_ca_data(df, sp):
    df = add_region_category(df)
    df = df[df['size'] == 'large']
    df = df[df['time (s)'] > 260]
    df = df[df['dist_c'] < 60]
    data = {
    'path' : [], 
    'particle' : [], 
    'treatment' : [], 
    'region' : [],
    'tracked' : [], 
    'corrected calcium' : []
        }
    gb = ['path', 'particle', 'treatment', 'region', 'tracked']
    for k, grp in df.groupby(gb):
        for i, col in enumerate(gb):
            data[col].append(k[i])
        data['corrected calcium'].append(grp['ca_corr'].mean())
    data = pd.DataFrame(data)
    data.to_csv(sp)
    return data

def ca_tracked_untracked_barplot(data):
    fig, axs = plt.subplots(1, 2, sharey=True)
    ax0, ax1 = axs.ravel()
    data0 = data[data['tracked'] == True]
    order = ('center', 'anterior', 'lateral', 'posterior')
    sns.boxenplot(data=data0, x='region', y='corrected calcium', hue='treatment', 
                hue_order=('DMSO (MIPS)', 'MIPS'), palette=pal1, ax=ax0, order=order)
    #sns.stripplot(data=data0, x='region', y='corrected calcium', hue='treatment', 
        #          hue_order=('DMSO (MIPS)', 'MIPS'), dodge=True, 
       #           palette=pal1, ax=ax0)
    data1 = data[data['tracked'] == False]
    sns.boxenplot(data=data1, x='region', hue='treatment', y='corrected calcium',
                hue_order=('DMSO (MIPS)', 'MIPS'), palette=pal1, ax=ax1, order=order)
    #sns.stripplot(data=data1, x='region', hue='treatment', y='corrected calcium',
     #             hue_order=('DMSO (MIPS)', 'MIPS'), dodge=True, 
      #            palette=pal1, ax=ax1)
    #ax0.set_yscale('log')
    #ax1.set_yscale('log')
    plt.show()


def get_high_calcium_platelets(df, cadf, save_path):
    idxs = cadf.index.values
    ca_vals = cadf['corrected calcium'].values
    where_high = np.where(ca_vals > 10)
    cadf = cadf.loc[idxs[where_high], :]
    cadf = cadf.set_index(['path', 'particle'])
    idxs = cadf.index.values
    df = df.set_index(['path', 'particle'])
    df = df.loc[idxs, :]
    for tx, grp in df.groupby('treatment'):
        print(tx)
        for p, g in grp.groupby('path'):
            print(p, ': ', len(g))
    df.to_csv(save_path)
    return df


def sample_platelets(df, save_path, treatments=('DMSO (MIPS)', 'MIPS'), n_each=50):
    df = df.set_index(['path', 'particle'])
    out = []
    for tx in treatments:
        sdf = df[df['treatment'] == tx]
        idxs = sdf.index.values
        chosen = np.random.choice(idxs, n_each)
        cdf = sdf.loc[chosen, :]
        out.append(cdf)
    out = pd.concat(out)
    out.to_csv(save_path)
    return out


def add_rolled_calcium(data):
    data = data.sort_values('time (s)')
    for k, grp in data.groupby(['path', 'particle']):
        idxs = grp.index.values
        rolled = grp.ca_corr.rolling(window=30).mean()
        data.loc[idxs, 'corrected calcium (AU)'] = rolled
    return data 


def add_peak_heigh_measure(data):
    data = add_region_category(data)
    data = data.sort_values('time (s)')
    for k, grp in data.groupby(['path', 'particle']):
        idxs = grp.index.values
        ca = grp['corrected calcium (AU)'].values
        t = grp['time (s)'].values
        t_diff = t[-1] - t[0]
        data.loc[idxs, 't_diff'] = t_diff
        peaks = find_peaks(ca)[0]
        n_peaks = len(peaks)
        freq = n_peaks / t_diff
        data.loc[idxs, 'ca_freq'] = freq
        prom = peak_prominences(ca, peaks)[0]
        mean = np.nanmean(ca)
        mean_prom = prom.mean()
        data.loc[idxs, 'mean_ca_peak_size'] = mean_prom
        mean_mns_prom = mean - mean_prom
        data.loc[idxs, 'mean_mns_prom'] = mean_mns_prom
        prom_ratio = np.log10(mean_mns_prom / mean_prom)
        data.loc[idxs, 'mean_to_peak_ratio'] = prom_ratio
        growth = grp[grp['time (s)'] < 270]
        data.loc[idxs, 'growth_mean_ca'] = np.nanmean(growth['corrected calcium (AU)'])
        consol = grp[grp['time (s)'] > 270]
        data.loc[idxs, 'consol_mean_ca'] = np.nanmean(consol['corrected calcium (AU)'])
        data.loc[idxs, 'time_start'] = t
        data.loc[idxs, 'start_region'] = grp['region'].values[0]
    return data


def plot_calcium_over_time(data):
    fig, axs = plt.subplots(1, 2)
    ax0, ax1 = axs.ravel()
    d0 = data[data['treatment'] == 'DMSO (MIPS)']
    sns.lineplot(data=d0, x='time (s)', y='corrected calcium (AU)', hue='mean_to_peak_ratio', ax=ax0)
    d1 = data[data['treatment'] == 'MIPS']
    sns.lineplot(data=d1, x='time (s)', y='corrected calcium (AU)', hue='mean_to_peak_ratio', ax=ax1)
    plt.show()



def collect_ca_points(data, save_path):
    gb = ['path', 'particle', 'treatment']
    vars = ['t_diff', 'time_start', 'ca_freq', 
            'mean_ca_peak_size', 'mean_mns_prom', 'mean_to_peak_ratio', 
            'growth_mean_ca', 'consol_mean_ca', 'start_region']
    out = {}
    for c in gb:
        out[c] = []
    for c in vars:
        out[c] = []
    for k, grp in data.groupby(gb):
        for i, c in enumerate(gb):
            out[c].append(k[i])
        for c in vars:
            out[c].append(grp[c].values[0])
    out = pd.DataFrame(out)
    out.to_csv(save_path)
    return out


def ca_scatter(x, y, hue, data, hue_order):
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    ax0, ax1 = axs.ravel()
    d0 = data[data['treatment'] == 'DMSO (MIPS)']
    sns.scatterplot(data=d0, x=x, y=y, hue=hue, ax=ax0, hue_order=hue_order)
    d1 = data[data['treatment'] == 'MIPS']
    sns.scatterplot(data=d1, x=x, y=y, hue=hue, ax=ax1, hue_order=hue_order)
    plt.show()


d = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes'
#file_names = ('211206_mips_df.parquet', '211206_veh-mips_df.parquet', '230301_MIPS_and_DMSO.parquet')
#file_paths = [os.path.join(d, n) for n in file_names]
#df = add_basic_variables_to_files(file_paths)
#df['tracked'] = df['nrtracks'] > 1
#sp = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/calcium/tracked_vs_untracked_region.csv'
#data = get_tracked_untracked_ca_data(df, sp)
#ca_tracked_untracked_barplot(data)


#data = pd.read_csv(sp)
#data = data[data['tracked'] == True]
#data['corrected calcium'].plot(kind='hist', bins=50, logy=True)
#plt.show()
save_path = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/calcium/gt10_calcium_platelet_data_MIPS-and-veh.csv'
#get_high_calcium_platelets(df, data, save_path)
#high_ca = pd.read_csv(save_path)
#sp1 = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/calcium/gt60_calcium_platelet_data_MIPS-and-veh_500each.csv'
#sample_platelets(high_ca, sp1, treatments=('DMSO (MIPS)', 'MIPS'), n_each=500)
#data = pd.read_csv(save_path)
#data = add_rolled_calcium(data)
#data = add_peak_heigh_measure(data)
#data = data[data['t_diff'] > 60]
#plot_calcium_over_time(data)
save_points = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/calcium/high_platlet_points_MIPS-and-veh.csv'
#data = collect_ca_points(data, save_points)
data = pd.read_csv(save_points)
#data = data[data['start_region'] != 'center']
data = data[data['time_start'] > 270]
#data = data[data['t_diff'] < 60]
#ca_scatter('consol_mean_ca', 'mean_to_peak_ratio', 'growth_mean_ca', data)
ca_scatter('consol_mean_ca', 'mean_to_peak_ratio', 'start_region', data, ('center', 'anterior', 'lateral', 'posterior'))
ca_scatter('consol_mean_ca', 'mean_to_peak_ratio', 'time_start', data, None)
ca_scatter('consol_mean_ca', 'growth_mean_ca', 'mean_to_peak_ratio', data, None)
ca_scatter('consol_mean_ca', 'growth_mean_ca', 'start_region', data, ('center', 'anterior', 'lateral', 'posterior'))
ca_scatter('consol_mean_ca', 'growth_mean_ca', 'time_start', data, None)
ca_scatter('time_start', 'consol_mean_ca', 'start_region', data, ('center', 'anterior', 'lateral', 'posterior'))
# diff between mean and average peak height (mean - peak height = neg means fewer large peaks, pos means high stable)