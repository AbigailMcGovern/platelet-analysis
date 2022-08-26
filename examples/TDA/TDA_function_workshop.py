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
    out = get_longest_loop_data(df, save_path, centile=75, col='nd15_percentile', y_col='ys_pcnt', x_col='x_s_pcnt')
    out.to_csv(save_path)
    plot_averages(out, units)


def get_longest_loop_data(df, save_path, centile=75, col='nd15_percentile', y_col='ys_pcnt', x_col='x_s_pcnt'):
    data = df[df[col] > centile]
    data = data[['frame', x_col, y_col, 'path']]
    paths = pd.unique(data['path'])
    path = []
    frames = []
    deaths = []
    births = []
    lifespan = []
    outlierness = []
    for inj in paths:
        inj_data = data[data['path'] == inj]
        out_dict = get_outlier_info_for_data(inj_data, x_col, y_col)
        path = path + [inj, ] * len(out_dict['frame'])
        frames = frames + out_dict['frame']
        deaths = deaths + out_dict['deaths']
        births = births + out_dict['births']
        lifespan = lifespan + out_dict['lifespan']
        outlierness = outlierness + out_dict['outlierness']
    out = {
        'frame' : frames, 
        'births' : births, 
        'deaths' : deaths, 
        'outlierness' : outlierness, 
        'lifespan' : lifespan
    }
    out = pd.DataFrame(out)
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
            pdf = df[df['path'] == p]
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


def max_loop_over_time_comparison(paths, save_path, centile=75, col='nd15_percentile', y_col='ys_pcnt', x_col='x_s_pcnt', units='%'):
    loop_data = {}
    for p in paths:
        df = pd.read_csv(p)
        out = get_longest_loop_data(df, save_path, centile=centile, col=col, y_col=y_col, x_col=x_col)
        key = Path(p).stem
        loop_data[key] = out
    l_data, o_data, x, y0, y1, hue = _prep_for_seborn(loop_data, units)
    _make_comparative_plots(l_data, o_data, x, y0, y1, hue)



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
    for inh in loop_data:
        time = loop_data[inh]['frame'].values / 0.321764322705706
        lifespan = loop_data[inh]['lifespan'].values
        outlierness = loop_data[inh]['outlierness'].values
        l_data[x] = np.concatenate([l_data[x], time])
        o_data[x] = np.concatenate([o_data[x], time])
        l_data[y0] = np.concatenate([l_data[x], lifespan])
        o_data[y1] = np.concatenate([o_data[y1], outlierness])
        tx_name = get_treatment_name(inh)
        l_data[hue] = np.array([tx_name, ] * len(time))
        o_data[hue] = np.array([tx_name, ] * len(time))
    l_data = pd.DataFrame(l_data)
    o_data = pd.DataFrame(o_data)
    return l_data, o_data, x, y0, y1, hue



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
    elif 'mips' in inh:
        out = 'MIPS'
    elif 'sq' in inh:
        out = 'SQ'
    else:
        out = inh
    return out




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
    df = pd.read_parquet(sp)

    # this experiment forms a figure 8 for some reason... might be a laser problem but might not... interesting but grounds for exclusion
    df = df[df['path'] != '191128_IVMTR33_Inj5_saline_exp3']

    # ---------------------------------------
    # Generate 75th centile scatter animation
    # ---------------------------------------
    #save_path = '/Users/amcg0011/Data/platelet-analysis/TDA/saline_QNxy_99thcentile_density.gif'
    #anim = density_centile_scatter_animation(df, save_path, centile=99, y_col='ys_pcnt', x_col='x_s_pcnt', ylim=(0, 100), xlim=(0, 100))

    #save_path = '/Users/amcg0011/Data/platelet-analysis/TDA/saline_QNxy_75thcentile_density_persistance.gif'
    #anim = density_centile_persistance_animation(df, save_path, centile=75, col='nb_density_15_pcntf', y_col='ys_pcnt', x_col='x_s_pcnt')

    paths = pd.unique(df['path'])
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


    persistance_diagrams_for_timepointz(df, centile=75, col='nb_density_15_pcntf', 
                                        y_col='ys_pcnt', x_col='x_s_pcnt', units='%',
                                        path='200527_IVMTR73_Inj4_saline_exp3', tps=(28, 115, 190))

    
#df80 = df[df['nd15_percentile'] > 80]
#paths = pd.unique(df80['path'])
# df = quantile_normalise_variables(df, vars=('x_s', 'ys'))

# sp = '/Users/amcg0011/Data/platelet-analysis/dataframes/211206_saline_df_220818_amp0.parquet'