import enum
from tkinter import N
from turtle import right
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import seaborn as sns
from scipy.stats import iqr, scoreatpercentile
from pathlib import Path
from scipy import stats

# ---------------------------------------
# Generate 75th centile scatter animation
# ---------------------------------------


def density_centile_scatter_animation(df, save_path, centile=99, col='nd15_percentile', y_col='ys', x_col='x_s', xlim =(-50, 50), ylim =(-90, 50)):
    fig = plt.figure(figsize=(5, 7))
    axis = plt.axes(xlim=xlim, 
                    ylim=ylim) 
    scatter = axis.scatter([], [], s=2) 

    data = df[df[col] > centile]
    data = data[['frame', x_col, y_col]]

    def init(): 
        scatter.set_offsets([[],])
        return scatter, 

    def animate(i):
        frame_data = data[data['frame'] == i]
        x = frame_data[x_col].values
        y = frame_data[y_col].values
        scatter.set_offsets(np.c_[x, y])
        #axis.scatter(x, y)
        return scatter, 

    anim = FuncAnimation(fig, animate, frames=193, interval=200, blit=True)
   
    anim.save(save_path, # save with mp4 ext
            fps = 30) # writer = 'ffmpeg'
    
    return anim



# ---------------------------------------
# Generate persistance diagram animations
# ---------------------------------------

def density_centile_persistance_animation(df, save_path, centile=75, col='nd15_percentile', y_col='ys', x_col='x_s'):
    fig = plt.figure()
    axis = plt.axes(xlim =(0, 80), 
                    ylim =(0, 80)) 
    scatter_0 = axis.scatter([], [], s=3, c='blue') 
    scatter_1 = axis.scatter([], [], s=3, c='orange') 
    h1_line = axis.axline((0, 0), (1, 1), ls='--', c='black')

    data = df[df[col] > centile]
    data = data[['frame', x_col, y_col]]
    
    def init():
        scatter_0.set_data([], [])
        scatter_1.set_data([], [])
        
    def animate_persistance(i):
        frame_data = data[data['frame'] == i]
        X = frame_data[[x_col, y_col]].values
        if len(X) > 0:
            dgms = ripser(X)['dgms']
            h0 = dgms[0]
            h1 = dgms[1]
            scatter_0.set_offsets(np.c_[h0[:, 0], h0[:, 1]])
            scatter_1.set_offsets(np.c_[h1[:, 0], h1[:, 1]])
            h0_max = np.max(h0)
            h0_line = axis.axhline(h0_max, ls='--')
        else:
            h0_line = axis.axhline(0, ls='--')
        return scatter_0, scatter_1, h0_line, 

    anim = FuncAnimation(fig, animate_persistance, frames=193, interval=200, blit=True)
   
    anim.save(save_path, fps = 30) #writer = 'ffmpeg'

    return anim


def plot_longest_loop_all(df, save_path, centile=75, col='nd15_percentile', y_col='ys_pcnt', x_col='x_s_pcnt', units='%'):
    frames = list(range(df['frame'].max()))
    data = df[df[col] > centile]
    data = data[['frame', x_col, y_col]]
    deaths = []
    births = []
    lifespan = []
    outlierness = []
    with tqdm(total=len(frames)) as progress:
        for t in frames:
            data_t = data[data['frame'] == t]
            X = data_t[[x_col, y_col]].values
            if len(X) > 0:
                dgms = ripser(X)['dgms']
                h1 = dgms[1]
                #print(h1)
                if len(h1) > 0:
                    diff = h1[:, 1] - h1[:, 0]
                    i = np.argmax(diff)
                    std = np.std(diff)
                    mean = np.mean(diff)
                    max = diff[i]
                    std_from_mean = (max - mean) / std
                    outlierness.append(std_from_mean)
                    lifespan.append(diff[i])
                    deaths.append(h1[i, 1])
                    births.append(h1[i, 0])
                else:
                    lifespan.append(np.NaN)
                    births.append(np.NaN)
                    deaths.append(np.NaN)
                    outlierness.append(np.NaN)
            else:
                lifespan.append(np.NaN)
                births.append(np.NaN)
                deaths.append(np.NaN)
                outlierness.append(np.NaN)
            progress.update(1)  
    lifespan = np.array(lifespan)
    births = np.array(births)
    deaths = np.array(deaths)
    outlierness = np.array(outlierness)
    out = {
        'frame' : frames, 
        'births' : births, 
        'deaths' : deaths, 
        'outlierness' : outlierness, 
        'lifespan' : lifespan
    }
    out = pd.DataFrame(out)
    out.to_csv(save_path)
    plot_barcode_over_time(out, units)
    
    
def plot_barcode_over_time(data, units):
    lifespan = data['lifespan']
    births = data['births']
    deaths = data['deaths']
    outlierness = data['outlierness']
    frames = data['frame']
    fig, axes = plt.subplots(3, 1, sharex=True)
    axes = axes.ravel()
    # TOP PLOT - Barcode births and deaths
    ax0 = axes[0]
    ax0.plot(frames, births, label='birth')
    ax0.plot(frames, deaths, label='death')
    ax0.set_ylabel(f'Radius around point ({units})')
    ax0.legend()
    # MIDDLE PLOT - Barcode lifespan
    ax1 = axes[1]
    ax1.plot(frames, lifespan, label='Lifespan')
    ax1.set_ylabel(f'Radius around point ({units})')
    ax1.legend()
    # BOTTOM PLOT - Barcode lifespan number of standard deviations from mean
    ax2 = axes[2]
    ax2.plot(frames, outlierness, label='Outlierness')
    ax2.set_ylabel(f'Standard deviations from mean ({units})')
    ax2.legend()
    plt.show()


def get_outlier_info_for_data(data, x_col, y_col):
    frames = list(range(data['frame'].max()))
    deaths = []
    births = []
    lifespan = []
    outlierness = []
    path = []
    for t in frames:
        data_t = data[data['frame'] == t]
        X = data_t[[x_col, y_col]].values
        if len(X) > 0:
            dgms = ripser(X)['dgms']
            h1 = dgms[1]
            #print(h1)
            if len(h1) > 0:
                diff = h1[:, 1] - h1[:, 0]
                i = np.argmax(diff)
                std = np.std(diff)
                mean = np.mean(diff)
                max = diff[i]
                std_from_mean = (max - mean) / std
                outlierness.append(std_from_mean)
                lifespan.append(diff[i])
                deaths.append(h1[i, 1])
                births.append(h1[i, 0])
            else:
                lifespan.append(np.NaN)
                births.append(np.NaN)
                deaths.append(np.NaN)
                outlierness.append(np.NaN)
        else:
            lifespan.append(np.NaN)
            births.append(np.NaN)
            deaths.append(np.NaN)
            outlierness.append(np.NaN)
    out = {
        'frame' : frames, 
        'births' : births, 
        'deaths' : deaths, 
        'outlierness' : outlierness, 
        'lifespan' : lifespan
    }
    return out


def plot_longest_loop_average(df, save_path, centile=75, col='nd15_percentile', y_col='ys_pcnt', x_col='x_s_pcnt', units='%'):
    out = get_longest_loop_data(df, centile=75, col='nd15_percentile', y_col='ys_pcnt', x_col='x_s_pcnt')
    out.to_csv(save_path)
    plot_averages(out, units)


def get_longest_loop_data(df, centile=75, col='nb_density_15_pcntf', y_col='ys_pcnt', x_col='x_s_pcnt', get_accessory_data=False):
    data = df[df[col] > centile]
    data = data[['frame', x_col, y_col, 'path']]
    injuries = pd.unique(data['path'])
    frames = []
    deaths = []
    births = []
    lifespan = []
    outlierness = []
    paths = []
    tx_name = get_treatment_name(data['path'].values[0])
    with tqdm(desc=f'Getting max barcode data for treatment = {tx_name}', total=len(paths)) as progress:
        for inj in injuries:
            inj_data = data[data['path'] == inj]
            out_dict = get_outlier_info_for_data(inj_data, x_col, y_col)
            #out_dict['path'] = [inj, ] * len()
            paths_new = [inj, ] * len(out_dict['frame'])
            paths = paths + paths_new
            frames = frames + out_dict['frame']
            deaths = deaths + out_dict['deaths']
            births = births + out_dict['births']
            lifespan = lifespan + out_dict['lifespan']
            outlierness = outlierness + out_dict['outlierness']
            progress.update(1)
    out = {
        'frame' : frames, 
        'births' : births, 
        'deaths' : deaths, 
        'outlierness' : outlierness, 
        'lifespan' : lifespan, 
        'path' : paths, 
    }
    out = pd.DataFrame(out)
    if get_accessory_data:
        uframes = pd.unique(out['frame'])
        upaths = pd.unique(out['path'])
        its = len(upaths) * len(uframes)
        with tqdm(desc=f'Getting accessory data for treatment = {tx_name}', total=its) as progress:
            for p in upaths:
                p_df = df[df['path'] == p]
                p_out = out[out['path'] == p]
                for f in uframes:
                    f0_df = p_df[p_df['frame'] == f]
                    f_out = p_out[p_out['frame'] == f]
                    idx = f_out.index.values
                    count0 = get_count(f0_df, thresh_col=col)
                    out.loc[idx, 'count'] = count0
                    f1_df = p_df[p_df['frame'] == f + 1]
                    if len(f1_df) > 0:
                        count1 = get_count(f1_df, thresh_col=col)
                        turnover = ((count1 - count0) / count0) * 100
                    else:
                        turnover = np.NaN
                    out.loc[idx, 'turnover'] = turnover
                    out.loc[idx, 'dv (um/s)'] = f0_df['dv'].mean()
                    out.loc[idx, 'corrected calcium'] = f0_df['ca_corr'].mean()
                    out.loc[idx, 'density (platelets/um^2)'] = f0_df['nb_density_15'].mean()
                    progress.update(1)
    return out


def plot_averages(data, units):
    frame = data['frame']
    time = np.array(frame) / 0.321764322705706
    x = 'time (s)'
    y0 = f'radius {units}'
    hue = 'measure'
    # Data for plot 0
    # ---------------
    bd_time = np.concatenate([time, time.copy()])
    bd_radius = np.concatenate([data['births'], data['deaths']])
    birth = ['birth', ] * len(data['births'])
    death = ['death', ] * len(data['deaths'])
    bd_event = np.array(birth + death)
    bd_data = {
        x : bd_time, 
        y0 : bd_radius, 
        hue : bd_event
    }
    bd_data = pd.DataFrame(bd_data)
    # Data for plot 1
    # ---------------
    l_data = {
        x : time, 
        y0 : data['lifespan'], 
        hue : ['lifespan', ] * len(time)
    }
    l_data = pd.DataFrame(l_data)
    # Data for plot 2
    # ---------------
    y1 = 'Standard deviations from mean'
    o_data = {
        x : time, 
        y1 : data['outlierness'], 
        hue : ['outlierness', ] * len(time)
    }
    o_data = pd.DataFrame(o_data)
    _make_plot(x, y0, y1, hue, bd_data, l_data, o_data)


def _make_plot(x, y0, y1, hue, bd_data, l_data, o_data):
    sns.set_style("ticks")
    fig, axes = plt.subplots(3, 1, sharex=True)
    ax0, ax1, ax2 = axes.ravel()
    e0 = sns.lineplot(x, y0, data=bd_data, hue=hue, ax=ax0)
    e1 = sns.lineplot(x, y0, data=l_data, hue=hue, ax=ax1)
    e2 = sns.lineplot(x, y1, data=o_data, hue=hue, ax=ax2)
    #sns.despine()
    plt.show()


# ------------------
# Number of features
# ------------------

def number_outliying_over_time(df, save_path, centile=75, col='nb_density_15_pcntf', y_col='ys_pcnt', x_col='x_s_pcnt', units='%'):
    out = get_outlier_number(df, x_col, y_col, units)
    out = pd.DataFrame(out)
    out.to_csv(save_path)
    x = 'time (s)'
    y0 = f'upper bound {units}'
    y1 = 'n outliers'
    out['measure'] = ['n outliers', ] * len(out)
    y2 = 'max outlier / mean outlier'
    sns.set_style("ticks")
    fig, axes = plt.subplots(3, 1, sharex=True)
    ax0, ax1, ax2 = axes.ravel()
    e0 = sns.lineplot(x, y0, data=out, ax=ax0)
    e1 = sns.lineplot(x, y1, data=out, ax=ax1, hue='measure')
    e2 = sns.lineplot(x, y2, data=out, ax=ax2)
    plt.show()


def get_outlier_number(data, x_col, y_col, units):
    frames = np.arange(data['frame'].max())
    paths = pd.unique(data['path'])
    n_outliers = []
    outlier_upper = []
    outlier_max_div_mean = []
    time = frames / 0.321764322705706
    time = np.concatenate([time.copy() for _ in range(len(paths))])
    with tqdm(total=len(paths) * len(frames)) as progress:
        for p in paths:
            pdf = data[data['path'] == p]
            for t in frames:
                data_t = pdf[pdf['frame'] == t]
                X = data_t[[x_col, y_col]].values
                if len(X) > 0:
                    dgms = ripser(X)['dgms']
                    h1 = dgms[1]
                    #print(h1)
                    if len(h1) > 0:
                        diff = h1[:, 1] - h1[:, 0]
                        #IQR = iqr(diff)
                        #Q3 = scoreatpercentile(diff, 75)
                        #upper = Q3 + (1.5 * IQR)
                        mean = np.mean(diff)
                        std = np.std(diff)
                        upper  = mean + (std * 5)
                        idxs = np.where(diff > upper)[0]
                        n = len(idxs)
                        outliers = diff[idxs]
                        if n > 0:
                            i = np.argmax(outliers)
                            max = outliers[i]
                            outliers = np.delete(outliers, i)
                            if len(outliers) > 0:
                                o_mean = np.mean(outliers)
                                max_div_mean = max / o_mean # expect close to 1 on average in toroidal phase
                            else:
                                max_div_mean = 1
                        else: 
                            max_div_mean = 0 # really this is undefined, just want to know if the top outlier is very far from others
                        n_outliers.append(n)
                        outlier_upper.append(upper)
                        outlier_max_div_mean.append(max_div_mean)
                    else:
                        n_outliers.append(np.NaN)
                        outlier_upper.append(np.NaN)
                        outlier_max_div_mean.append(np.NaN)
                else:
                    n_outliers.append(np.NaN)
                    outlier_upper.append(np.NaN)
                    outlier_max_div_mean.append(np.NaN)
                progress.update(1)
    x = 'time (s)'
    out = {
        x : time, 
        f'upper bound {units}' : outlier_upper, 
        'n outliers' : n_outliers, 
        'max outlier / mean outlier' : outlier_max_div_mean, 
    }
    return out


# -------------------------------------
# Single timepoint persistance diagrams
# -------------------------------------

def persistance_diagrams_for_timepointz(
        df, 
        centile=75, 
        col='nb_density_15_pcntf', 
        y_col='ys_pcnt', 
        x_col='x_s_pcnt', 
        units='%',
        path='200527_IVMTR73_Inj4_saline_exp3', 
        tps=(10, 28, 115, 190)
        ):

    df = df[df['path'] == path]
    df = df[df[col] > centile]
    tp_dfs = {}
    x = f'bith radius {units}'
    y = f'death radius {units}'
    hue = 'homology dimension'
    size = 'Loop lifespan'
    for t in tps:
        tdf = df[df['frame'] == t]
        X = tdf[[x_col, y_col]].values
        dgms = ripser(X)['dgms']
        diff = dgms[1][:, 1] - dgms[1][:, 0]
        h0_size = [np.mean(diff), ] * len(dgms[0])
        data = {
            x : np.concatenate([dgms[0][:, 0], dgms[1][:, 0]]), 
            y : np.concatenate([dgms[0][:, 1], dgms[1][:, 1]]), 
            hue : ['H0', ] * len(dgms[0]) + ['H1', ] * len(dgms[1]), 
            size : np.concatenate([h0_size, diff])
        }
        data = pd.DataFrame(data)
        tp_dfs[t] = data
    n_graphs = len(tps)
    sns.set_style("ticks")
    fig, axes = plt.subplots(n_graphs, 1, sharex=True, sharey=True)
    max_size = 150
    max_diffs = {}
    mdl = []
    for key in tp_dfs.keys():
        plot_df = tp_dfs[key]
        m = plot_df[size].max()
        max_diffs[key] = m
        mdl.append(m)
    max_diff = np.max(mdl)
    max_sizes = {key : np.floor((max_diffs[key] / max_diff) * max_size) for key in  tp_dfs.keys()}

    for i, ax in enumerate(axes.ravel()):
        key = tps[i]
        plot_df = tp_dfs[key]
        ms = max_sizes[key]
        max = plot_df[plot_df[hue] == 'H0'][y].max()
        g = sns.scatterplot(x, y, hue=hue, data=plot_df, palette='rocket', ax=ax, size=size, sizes=(10, ms), edgecolor='black')
        ax.axline((0, 0), (1, 1), ls='--', c='black')
        ax.axhline(max, ls='--', c='black')
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    fig.subplots_adjust(right=0.6)
    plt.show()



# -----------------------------------------------
# Multiple group comparison max barcode over time
# -----------------------------------------------


def max_loop_over_time_comparison(paths, centile=75, col='nd15_percentile', y_col='ys_pcnt', x_col='x_s_pcnt', units='%'):
    loop_data = {}
    for p in paths:
        df = pd.read_parquet(p)
        out = get_longest_loop_data(df, centile=centile, col=col, y_col=y_col, x_col=x_col)
        key = Path(p).stem
        loop_data[key] = out
    l_data, o_data, x, y0, y1, hue = _prep_for_seborn(loop_data, units)
    _make_comparative_plots(l_data, o_data, x, y0, y1, hue)



def max_loop_over_time_data(paths, save_path, centile=75, col='nd15_percentile', y_col='ys_pcnt', x_col='x_s_pcnt', units='%'):
    loop_data = {}
    n = len(paths)
    for i, p in enumerate(paths):
        print(f'Obtaining data for dataframe {i} of {n}')
        print(f'path: {p}')
        df = pd.read_parquet(p)
        out = get_longest_loop_data(df, centile=centile, col=col, y_col=y_col, x_col=x_col, get_accessory_data=True)
        key = Path(p).stem
        loop_data[key] = out
    out_df = generate_full_data_sheet(loop_data, units)
    #spo = os.path.join(save_dir, save_name + '_outlierness.csv')
    out_df.to_csv(save_path)
    return out_df
    



def _prep_for_seborn(loop_data, units):
    x = 'time (s)'
    y0 = f'radius {units}'
    y1 = 'Standard deviations from mean'
    hue = 'treatment'
    l_data = {
        x : [], 
        y0 : [], 
        hue : []
    }
    o_data = {
        x : [], 
        y1 : [], 
        hue : []
    }
    for inh in loop_data.keys():
        time = loop_data[inh]['frame'].values / 0.321764322705706
        lifespan = loop_data[inh]['lifespan'].values
        outlierness = loop_data[inh]['outlierness'].values
        assert len(time) == len(lifespan)
        assert len(time) == len(outlierness)
        l_data[x] = np.concatenate([l_data[x], time])
        o_data[x] = np.concatenate([o_data[x], time])
        l_data[y0] = np.concatenate([l_data[y0], lifespan])
        o_data[y1] = np.concatenate([o_data[y1], outlierness])
        tx_name = get_treatment_name(inh)
        l_data[hue] = np.concatenate([l_data[hue], np.array([tx_name, ] * len(time))])
        o_data[hue] = np.concatenate([o_data[hue], np.array([tx_name, ] * len(time))])
    l_data = pd.DataFrame(l_data)
    o_data = pd.DataFrame(o_data)
    return l_data, o_data, x, y0, y1, hue


def generate_full_data_sheet(loop_data, units):
    x = 'time (s)'
    y0 = f'radius {units}'
    y1 = 'Standard deviations from mean'
    p = 'path'
    hue = 'treatment'
    df = { # should change this to defaltdict
        'treatment' : [], 
        'path' : [], 
        'frame' : [],
        'time (s)' : [], 
        f'radius {units}' : [], 
        'Standard deviations from mean' : [], 
        'turnover (%)' : [], 
        'platelet count' : [], 
        'dv (um/s)' : [], 
        'corrected calcium' : [], 
        'density (platelets/um^2)' : []
    }
    for inh in loop_data.keys():
        tx_name = get_treatment_name(inh)
        p = loop_data[inh]['path'].values
        tx = np.array([tx_name, ] * len(p))
        f = loop_data[inh]['frame'].values
        time = loop_data[inh]['frame'].values / 0.321764322705706
        lifespan = loop_data[inh]['lifespan'].values
        outlierness = loop_data[inh]['outlierness'].values
        turnover = loop_data[inh]['turnover'].values
        count = loop_data[inh]['count'].values
        dv = loop_data[inh]['dv (um/s)'].values
        ca_corr = loop_data[inh]['corrected calcium'].values
        dens = loop_data[inh]['density (platelets/um^2)'].values
        df['treatment'] = np.concatenate([df['treatment'], tx])
        df['path'] = np.concatenate([df['path'], p])
        df['frame'] = np.concatenate([df['frame'], f])
        df['time (s)'] = np.concatenate([df['time (s)'], time])
        df[f'radius {units}'] = np.concatenate([df[f'radius {units}'], lifespan])
        df['Standard deviations from mean'] = np.concatenate([df['Standard deviations from mean'], outlierness])
        df['turnover (%)'] = np.concatenate([df['turnover (%)'], turnover])
        df['platelet count'] = np.concatenate([df['platelet count'], count])
        df['dv (um/s)'] = np.concatenate([df['dv (um/s)'], dv])
        df['corrected calcium'] = np.concatenate([df['corrected calcium'], ca_corr])
        df['density (platelets/um^2)'] = np.concatenate([df['density (platelets/um^2)'], dens])
    df = pd.DataFrame(df)
    return df
        




def _make_comparative_plots(l_data, o_data, x, y0, y1, hue):
    sns.set_style("ticks")
    fig, axes = plt.subplots(2, 1, sharex=True)
    ax0, ax1 = axes.ravel()
    e1 = sns.lineplot(x, y0, data=l_data, hue=hue, ax=ax0)
    e2 = sns.lineplot(x, y1, data=o_data, hue=hue, ax=ax1)
    #sns.despine()
    plt.show()



def get_treatment_name(inh):
    if 'saline' in inh:
        out = 'saline'
    elif 'biva' in inh:
        out = 'bivalirudin'
    elif 'cang' in inh:
        out = 'cangrelor'
    elif 'veh-mips' in inh:
        out = 'MIPS vehicle'
    elif 'mips' in inh:
        out = 'MIPS'
    elif 'sq' in inh:
        out = 'SQ'
    else:
        out = inh
    return out

# '211206_veh-mips_df_220831.parquet'
def variable_versus_barcode_lifespan(df, var='platelet count', time_bins=((0, 60), (60, 180), (180, 600)), units='%'):
    tx_col = 'treatment'
    t_col = 'time (s)'
    l_col = f'radius {units}'
    o_col = 'Standard deviations from mean'
    sns.set_style("ticks")
    fig, axes = plt.subplots(len(time_bins), 2, sharey=True, sharex='col')
    for i, bin in enumerate(time_bins):
        t_min, t_max = bin
        data = df[(df[t_col] > t_min) & (df[t_col] < t_max)]
        data = injury_averages(data, units)
        if len(time_bins) == 1:
            ax0 = axes[0]
            ax1 = axes[1]
        else:
            ax0 = axes[i, 0]
            ax1 = axes[i, 1]
        slope, intercept = _stats_with_print(l_col, var, f'time bin: {bin} seconds | {var} vs lifespan', data)
        sns.scatterplot(x=l_col, y=var, hue=tx_col, data=data, ax=ax0) # palette="husl"
        sns.move_legend(ax0, "upper left", bbox_to_anchor=(1, 1))
        ax0.axline((0, intercept), (1, intercept + slope), ls='--', c='black')
        slope, intercept = _stats_with_print(o_col, var, f'time bin: {bin} seconds | {var} vs outlierness', data)
        sns.scatterplot(x=o_col, y=var, hue=tx_col, data=data, ax=ax1)
        sns.move_legend(ax1, "upper left", bbox_to_anchor=(1, 1))
        ax1.axline((0, intercept), (1, intercept + slope), ls='--', c='black')
    plt.show()



def count_versus_barcode_lifespan(df, time_bins=((0, 60), (60, 180), (180, 600)), units='%'):
    tx_col = 'treatment'
    p_col = 'path' 
    f_col = 'frame'
    t_col = 'time (s)'
    l_col = f'radius {units}'
    o_col = 'Standard deviations from mean'
    turn_col = 'turnover (%)'
    c_col = 'platelet count'
    sns.set_style("ticks")
    fig, axes = plt.subplots(len(time_bins), 2, sharey=True, sharex='col')
    for i, bin in enumerate(time_bins):
        t_min, t_max = bin
        data = df[(df[t_col] > t_min) & (df[t_col] < t_max)]
        data = injury_averages(data, units)
        ax0 = axes[i, 0]
        ax1 = axes[i, 1]
        slope, intercept = _stats_with_print(l_col, c_col, f'time bin: {bin} seconds | turnover vs lifespan', data)
        sns.scatterplot(x=l_col, y=c_col, hue=tx_col, data=data, ax=ax0) # palette="husl"
        sns.move_legend(ax0, "upper left", bbox_to_anchor=(1, 1))
        ax0.axline((0, intercept), (1, intercept + slope), ls='--', c='black')
        slope, intercept = _stats_with_print(l_col, c_col, f'time bin: {bin} seconds | turnover vs outlierness', data)
        sns.scatterplot(x=o_col, y=c_col, hue=tx_col, data=data, ax=ax1)
        sns.move_legend(ax1, "upper left", bbox_to_anchor=(1, 1))
        ax1.axline((0, intercept), (1, intercept + slope), ls='--', c='black')
    plt.show()


def injury_averages(df, units='%'):
    tx_col = 'treatment'
    p_col = 'path' 
    f_col = 'frame'
    t_col = 'time (s)'
    l_col = f'radius {units}'
    o_col = 'Standard deviations from mean'
    turn_col = 'turnover (%)'
    c_col = 'platelet count'
    dv_col = 'dv (um/s)'
    ca_col = 'corrected calcium'
    dens_col = 'density (platelets/um^2)'
    paths = pd.unique(df[p_col])
    df_av = {
        tx_col : [df[df[p_col] == p][tx_col].values[0] for p in paths], 
        p_col : [df[df[p_col] == p][p_col].values[0] for p in paths], 
        f_col : [df[df[p_col] == p][f_col].values[0] for p in paths], 
        t_col : [df[df[p_col] == p][t_col].mean() for p in paths], 
        l_col : [df[df[p_col] == p][l_col].mean() for p in paths], 
        o_col : [df[df[p_col] == p][o_col].mean() for p in paths],
        turn_col : [df[df[p_col] == p][turn_col].mean() for p in paths],  
        c_col : [df[df[p_col] == p][c_col].mean() for p in paths], 
        dv_col : [df[df[p_col] == p][dv_col].mean() for p in paths], 
        ca_col : [df[df[p_col] == p][ca_col].mean() for p in paths], 
        dens_col : [df[df[p_col] == p][dens_col].mean() for p in paths], 
    }
    df_av = pd.DataFrame(df_av)
    return df_av


# --------------------
# Additional functions
# --------------------


def get_count(df, thresh_col='nd15_percentile', threshold=25):
    sml_df = df[df[thresh_col] > threshold]
    count = len(sml_df)
    return count


def _stats_with_print(x_col, y_col, desc, data):
    x = data[x_col]
    y = data[y_col]
    print(f'Pearson R for {desc}')
    r, p = stats.pearsonr(x, y)
    print(f'r = {r}, p = {p}')
    print(f'Linear regression for {desc}')
    slope, intercept, r, p, se = stats.linregress(x, y)
    rsq = r ** 2
    print(f'slope = {slope}, intercept = {intercept}, r = {r}, r squared = {rsq}, se = {se}, p = {p}')
    return slope, intercept


# ------------
# Execute code
# ------------

if __name__ == '__main__':
    
    # ---------
    # Load data
    # ---------

    d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
    #saline_n = '211206_saline_df_220614-amp0.parquet'
    #path = os.path.join(d, saline_n)
    sp = '/Users/amcg0011/Data/platelet-analysis/dataframes/211206_saline_df_220818_amp0.parquet'
    sp = '/Users/amcg0011/Data/platelet-analysis/dataframes/211206_saline_df_220822_amp0.parquet'
    sp = '/Users/amcg0011/Data/platelet-analysis/dataframes/211206_saline_df_220827_amp0.parquet'
    #df = pd.read_parquet(sp)
    #print(df.columns.values)

    # this experiment forms a figure 8 for some reason... might be a laser problem but might not... interesting but grounds for exclusion
    #df = df[df['path'] != '191128_IVMTR33_Inj5_saline_exp3']

    # ---------------------------------------
    # Generate 75th centile scatter animation
    # ---------------------------------------
    #save_path = '/Users/amcg0011/Data/platelet-analysis/TDA/saline_QNxy_99thcentile_density.gif'
    #anim = density_centile_scatter_animation(df, save_path, centile=99, y_col='ys_pcnt', x_col='x_s_pcnt', ylim=(0, 100), xlim=(0, 100))

    #save_path = '/Users/amcg0011/Data/platelet-analysis/TDA/saline_QNxy_75thcentile_density_persistance.gif'
    #anim = density_centile_persistance_animation(df, save_path, centile=75, col='nb_density_15_pcntf', y_col='ys_pcnt', x_col='x_s_pcnt')

    #paths = pd.unique(df['path'])
    #for path in paths:
     #   pdf = df[df['path'] == path]
     ##   save_path = f'/Users/amcg0011/Data/platelet-analysis/TDA/density-percentile_saline/Injuries/xy_ppcnt/saline_QNxy_75thcentile_density_{path}.gif'
       # save_path = f'/Users/amcg0011/Data/platelet-analysis/TDA/density-percentile_saline/Injuries/xy_ppcnt/persistance-diagrams/saline_QNxy_75thcentileF_density_persistance_{path}.gif'
      #  anim = density_centile_scatter_animation(pdf, save_path, centile=75, col='nb_density_15_pcntf')
        #anim = density_centile_persistance_animation(pdf, save_path, centile=75, col='nb_density_15_pcntf', y_col='ys_pcnt', x_col='x_s_pcnt')
    
    #save_path = '/Users/amcg0011/Data/platelet-analysis/TDA/saline_QNxy_90thcentile_outlierBC-aggregate.csv'
    #plot_longest_loop_all(df, save_path, centile=90, col='nd15_percentile', y_col='ys_pcnt', x_col='x_s_pcnt', units='%')
    
    #save_path = '/Users/amcg0011/Data/platelet-analysis/TDA/density-percentile_saline/Injuries/h1_max_barcode_vs_time/saline_QNxy_70thcentileF_outlierBC-average.csv'
    #plot_longest_loop_average(df, save_path, centile=70 , col='nb_density_15_pcntf', y_col='ys_pcnt', x_col='x_s_pcnt', units='%')
    #data = pd.read_csv(save_path)
    #plot_averages(data, '%')
    #save_path = '/Users/amcg0011/Data/platelet-analysis/TDA/density-percentile_saline/Injuries/h1_max_barcode_vs_time/saline_QNxy_75thcentileF_n-outliers-average-5.csv'
    #number_outliying_over_time(df, save_path, centile=75, col='nb_density_15_pcntf', y_col='ys_pcnt', x_col='x_s_pcnt', units='%')


    #persistance_diagrams_for_timepointz(df, centile=75, col='nb_density_15_pcntf', 
      #                                  y_col='ys_pcnt', x_col='x_s_pcnt', units='%',
       #                                 path='200527_IVMTR73_Inj4_saline_exp3', tps=(28, 115, 190))
    names = ['211206_saline_df_220827_amp0.parquet', '211206_biva_df.parquet', '211206_cang_df.parquet', '211206_sq_df.parquet', '211206_mips_df_220818.parquet']
    paths = [os.path.join(d, n) for n in names]
    #max_loop_over_time_comparison(paths, centile=75, col='nb_density_15_pcntf', y_col='ys_pcnt', x_col='x_s_pcnt', units='%')
    save_dir = '/Users/amcg0011/Data/platelet-analysis/TDA/treatment_comparison'
    save_name = 'saline_biva_cang_sq_mips_PH-data-all.csv'
    save_path = os.path.join(save_dir, save_name)
    #max_loop_over_time_data(paths, save_path, centile=75, col='nb_density_15_pcntf', y_col='ys_pcnt', x_col='x_s_pcnt', units='%')
    df = pd.read_csv(save_path)
    #df_av = injury_averages(df, units='%')
    #count_versus_barcode_lifespan(df, time_bins=((0, 60), (60, 180), (180, 600)), units='%')
    #variable_versus_barcode_lifespan(df, var='turnover (%)', time_bins=((0, 60), (60, 180), (180, 600)), units='%')
    #variable_versus_barcode_lifespan(df, var='platelet count', time_bins=((0, 600), ), units='%')
    #variable_versus_barcode_lifespan(df, var='dv (um/s)', time_bins=((0, 60), (60, 180), (180, 600)), units='%')
    #variable_versus_barcode_lifespan(df, var='density (platelets/um^2)', time_bins=((0, 60), (60, 180), (180, 600)), units='%')
    #variable_versus_barcode_lifespan(df, var='dv (um/s)', time_bins=((0, 600), ), units='%')
    #df = df[df['treatment'] == 'saline']
    #variable_versus_barcode_lifespan(df, var='corrected calcium', time_bins=((0, 60), (60, 180), (180, 600)), units='%')
    #variable_versus_barcode_lifespan(df, var='corrected calcium', time_bins=((0, 600), ), units='%')
    #variable_versus_barcode_lifespan(df, var='density (platelets/um^2)', time_bins=((0, 600), ), units='%')




#df80 = df[df['nd15_percentile'] > 80]
#paths = pd.unique(df80['path'])
# df = quantile_normalise_variables(df, vars=('x_s', 'ys'))

# sp = '/Users/amcg0011/Data/platelet-analysis/dataframes/211206_saline_df_220818_amp0.parquet'
