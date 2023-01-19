import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.signal import find_peaks



def find_max_donut_time(df, sn=None, sd=None, p='path', tx='treatment', t='time (s)', y='Standard deviations from mean', lose=1, thresh=6):
    #adf = find_smoothed_average(df, tx, t, y)
    df = rolling_variable(df, p=p, t=t, y=y)
    y = y + ' rolling'
    summary = {
        tx :[],
        f'max {y} mean' : [], 
        f'max {y} SEM' : [], 
        f'{t} mean': [], 
        f'{t} SEM': [],
        'frames mean' : [], 
        'frames SEM' : [],
        f'time to minus {lose} mean': [], 
        f'time to minus {lose} SEM': [], 
        f'time over {thresh} mean' : [], 
        f'time over {thresh} SEM' : [], 
    }
    result = {
        p : [],
        tx :[], 
        f'max {y}' : [],
        t : [], 
        'frames' : [],
        f'time to minus {lose}' : [], 
        f'time over {thresh}' : [], 
    }
    for k, g in df.groupby([p, tx, ]):
        g = g.sort_values(t)
        peaks, props = find_peaks(g[y].values)
        idx = np.min(peaks)
        mv = g[y].values[idx]
        #mv = g[y].max()
        #idx = np.argmax(g[y].values)
        time = g[t].values[idx]
        result[p].append(k[0])
        result[tx].append(k[1])
        result[t].append(time)
        result[f'max {y}'].append(mv)
        f = np.round(time * 0.321764322705706).astype(int)
        result['frames'].append(f)
        # get time to lose 2
        smlg = g[g[t] > time]
        val = mv - lose
        i = np.where(smlg[y].values < val)
        if len(i[0]) > 0:
            min_i = np.min(i)
            ttm = smlg[t].values[min_i] - time
        else:
            ttm = np.inf
        result[f'time to minus {lose}'].append(ttm)
        # get time over 5
        ts = pd.unique(g[t])
        interval = ts[1] - ts[0]
        i = np.where(g[y] > thresh)
        to = len(i[0]) * interval
        result[f'time over {thresh}'].append(to)
    result = pd.DataFrame(result)
    for k, g in result.groupby([tx, ]):
        summary[tx].append(k)
        summary[f'max {y} mean'].append(g[f'max {y}'].mean())
        summary[f'max {y} SEM'].append(g[f'max {y}'].sem())
        summary[f'{t} mean'].append(g[t].mean())
        summary[f'{t} SEM'].append(g[t].sem())
        summary['frames mean'].append(g['frames'].mean())
        summary['frames SEM'].append(g['frames'].sem())
        summary[f'time to minus {lose} mean'].append(g[f'time to minus {lose}'].mean())
        summary[f'time to minus {lose} SEM'].append(g[f'time to minus {lose}'].sem())
        summary[f'time over {thresh} mean'].append(g[f'time over {thresh}'].mean())
        summary[f'time over {thresh} SEM'].append(g[f'time over {thresh}'].sem())
    if sn is not None and sd is not None:
        sp0 = os.path.join(sd, sn +'_result.csv')
        result.to_csv(sp0)
        sp1 = os.path.join(sd, sn +'_summary.csv')
        summary = pd.DataFrame(summary)
        summary.to_csv(sp1)
    return result


def plot_max_donut_data(df):
    fig, axes = plt.subplots(3, 1, sharex=True)
    ax0, ax1, ax2 = axes.ravel()
    sns.scatterplot(x='max Standard deviations from mean rolling', y='time (s)', data=df, ax=ax0, hue='treatment')
    sns.move_legend(ax0, "upper left", bbox_to_anchor=(1, 1))
    sns.scatterplot(x='max Standard deviations from mean rolling', y='time to minus 1', data=df, ax=ax1, hue='treatment')
    sns.move_legend(ax1, "upper left", bbox_to_anchor=(1, 1))
    sns.scatterplot(x='max Standard deviations from mean rolling', y='time over 6', data=df, ax=ax2, hue='treatment')
    sns.move_legend(ax2, "upper left", bbox_to_anchor=(1, 1))
    plt.show()


def find_smoothed_average(df, tx='treatment', t='time (s)', y='Standard deviations from mean'):
    df = rolling_variable(df)
    t = t + ' rolling'
    adf = {
        t : [], 
        tx : [], 
        y : []
    }
    for k, grp in df.groupby([tx, t]):
        av = grp[y].mean()
        adf[t] = k[1]
        adf[tx] = k[0]
        adf[y] = av
    adf = pd.DataFrame(adf)
    return adf
    

def rolling_variable(df, p='path', t='time (s)', y='Standard deviations from mean', time_average=False):
    n = y + ' rolling'
    for k, g in df.groupby([p]):
        g = g.sort_values(t)
        idx = g.index.values
        rolling = g[y].rolling(window=20,win_type='bartlett',min_periods=3,center=True).mean()
        df.loc[idx, n] = rolling
    return df


def plot_donuts(df, x='time (s)', y='Standard deviations from mean rolling', hue='treatment'):
    sns.set_style("ticks")
    fig, ax = plt.subplots(1, 1)
    e1 = sns.lineplot(x, y, data=df, hue=hue, ax=ax)
    plt.show()


if __name__ == "__main__":
    d = '/Users/amcg0011/Data/platelet-analysis/TDA/treatment_comparison'
    n = 'saline_biva_cang_sq_mips_PH-data-all.csv'
    n = 'donuts_1_result.csv'
    p = os.path.join(d, n)
    df = pd.read_csv(p)
    #df = rolling_variable(df)
    #plot_donuts(df)
    #find_max_donut_time(df, 'donuts_1', d)
    plot_max_donut_data(df)