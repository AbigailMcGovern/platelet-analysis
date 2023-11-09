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
from plateletanalysis.variables.measure import quantile_normalise_variables_frame


# ---------------------------------------
# Generate 75th centile scatter animation
# ---------------------------------------


def density_centile_scatter_animation(df, save_path, centile=75, col='nd15_percentile', y_col='ys', x_col='x_s', xlim =(-50, 50), ylim =(-90, 50)):
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
    plt.rcParams['svg.fonttype'] = 'none'
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
    fig, axes = plt.subplots(1, n_graphs, sharex=True, sharey=True)
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
        #sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    #fig.subplots_adjust(right=0.6)
    plt.show()

