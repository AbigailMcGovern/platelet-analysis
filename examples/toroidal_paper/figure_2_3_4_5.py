import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plateletanalysis.variables.basic import time_seconds, get_treatment_name, tracking_time_var, add_region_category
from plateletanalysis.variables.neighbours import local_contraction, add_neighbour_lists, local_density
from plateletanalysis.variables.transform import cylindrical_coordinates, spherical_coordinates
from plateletanalysis.variables.measure import stability
from scipy import stats
from toolz import curry
from collections import defaultdict


def get_rs(lim=10, a=1000):
    r = 100
    rs = []
    while r > lim:
        rs.append(r)
        r = np.sqrt(r**2 - a/np.pi)
    rs.append(0)
    return rs[::-1]

iso_cylrs = get_rs()


def smooth_vars(df, vars, w=15, t='time (s)', gb=['path', 'particle'], add_suff=None):
    df = df.sort_values(t)
    for v in vars:
        if add_suff is not None:
            v_n = v + add_suff
        else:
            v_n = v
        for k, grp in df.groupby(gb):
            rolled = grp[v].rolling(window=w, center=True).mean()
            idxs = grp.index.values
            df.loc[idxs, v_n] = rolled
    return df


def local_densification(df):
    df = df.sort_values('time (s)')
    for k, grp in df.groupby(['path', 'particle']):
        idx = grp.index.values
        vals = np.diff(grp['nb_density_15'].values)
        vals = np.concatenate([[np.nan, ], vals])
        vals = vals * 0.32
        df.loc[idx, 'densification (/um3/sec)'] = vals
    df = smooth_vars(df, ['densification (/um3/sec)', ], 
                     w=15, t='time (s)', gb=['path', 'particle'], 
                     add_suff=None)
    return df

def plot_corrs(
        df, 
        x, 
        y, 
        bin_col_0='time_bin', 
        bin_order_0=['0-1200 s', '120-300 s', '300-600 s'], 
        bin_col_1='track_bin',
        bin_order_1=['1-15', '15-30', '30-194']
    ):
    plt.rcParams['svg.fonttype'] = 'none'
    func = get_var_data(x, y)
    data = df.groupby(['particle', 'path', bin_col_0, bin_col_1]).apply(func).reset_index()
    fig, axs = plt.subplots(len(bin_order_1), len(bin_order_0))
    axs = axs.ravel()
    i = 0
    for k, grp in data.groupby([bin_col_0, bin_col_1]):
        ax = axs[i]
        i+=1
        sns.scatterplot(data=grp, x=x, y=y, ax=ax, hue='nrtracks', palette="rocket_r", alpha=0.5)
        sns.despine(ax=ax)
        ax.set_title(f'{k[0]}: {k[1]}')
    fig.set_size_inches(9, 7)
    fig.subplots_adjust(right=0.97, left=0.09, bottom=0.12, top=0.9, wspace=0.4, hspace=0.4)
    plt.show()


def add_binns(
        df, 
        lbs=[0, 120, 300, 600], 
        ubs=[120, 300, 600, 1200], 
        lbs_cyl=None, 
        ubs_cyl=None
        ):
    t_bin = time_bin(lbs, ubs)
    df['time_bin'] = df['time (s)'].apply(t_bin)
    bins = ((1, 15), (15, 30), (30, 194))
    bin_func = bin_by_trk(bins)
    df['track_bin'] = df['nrtracks'].apply(bin_func)
    df = cylindrical_coordinates(df)
    if lbs_cyl is None or ubs_cyl is None:
        lbs_cyl = iso_cylrs[0:-1]
        ubs_cyl = iso_cylrs[1:]
    c_bin = cyl_bin(lbs_cyl, ubs_cyl)
    df['cyl_r_bin'] = df['cyl_radial'].apply(c_bin)
    bins_1 = ((1, 15), (15, 30), (30, 194))
    df['trackn_bin'] = df['tracknr'].apply(bin_func)
    return df


@curry
def get_var_data(x, y, grp):
    out = pd.DataFrame({
        x : [grp[x].mean(), ], 
        y : [np.nanmean(grp[y].values), ], 
        'nrtracks' : [grp['nrtracks'].mean(), ]
    })
    return out

@curry
def time_bin(lbs, ubs, t):
    #lbs = [0, 60, 120, 300, 600]
    #ubs = [60, 120, 300, 600, 1200]
    #lbs = [0, 30, 300]
    #ubs = [30, 300, 1200]
    for l, u in zip(lbs, ubs):
        if t >= l and t < u:
            return f'{l}-{u} s'
@curry    
def cyl_bin(lbs, ubs, t):
    #lbs = np.linspace(0, 100, 25)[0:-1]
    #ubs = np.linspace(0, 100, 25)[1:]
    for l, u in zip(lbs, ubs):
        if t >= l and t < u:
            return l + 0.5 * (u - l)
        
@curry
def bin_by_trk(bins, val):
    for l, u in bins:
        if val > l and val <= u:
            n = f"{l}-{u} s"
            return n


def add_psel_bin(vals):
    if vals.mean() > 544:
        return [True, ] * len(vals)
    else:
        return [False, ] * len(vals)
    
def psel_bin(df):
    print('psel bin')
    for k, grp in df.groupby(['path', 'particle']):
        idx = grp.index.values
        vals = add_psel_bin(grp['p-sel average intensity'].values)
        df.loc[idx, 'psel'] = vals
    #df['psel'] = df.groupby(['path', 'particle'])['p-sel average intensity'].apply(add_psel_bin)
    return df

def classify_exp_type(path):
    if path.find('exp5') != -1:
        return '10-20 min'
    elif path.find('exp3') != -1:
        return '0-10 min'
    else:
        return 'other'


def initial_plots(df, var, nrlim=10):
    fig, ax = plt.subplots(1, 1)
    data = df.groupby(['path', 'time (s)', 'psel'])[var].mean().reset_index()
    sns.lineplot(data=data, x='time (s)', y=var, hue='psel', ax=ax,) # err_style='bars', markers='o')
    sns.despine()
    ax.set_xlim(0, nrlim)
    plt.show()


def initial_plots_all(df, v0, v1, lim=180):
    fig, axs = plt.subplots(2, 1, sharex=True)
    ax0 = axs[0]
    ax1 = axs[1]
    plt.rcParams['svg.fonttype'] = 'none'
    #data0 = df.groupby(['path', 'time (s)', 'psel'])[v0].mean().reset_index()
    #data1 = df.groupby(['path', 'time (s)', 'psel'])[v1].mean().reset_index()
    sns.lineplot(data=df, x='time (s)', y=v0, hue='psel', ax=ax0, palette='rocket')
    sns.despine(ax=ax0)
    ax0.set_xlim(0, lim)
    sns.lineplot(data=df, x='time (s)', y=v1, hue='psel', ax=ax1, palette='rocket')
    sns.despine(ax=ax1)
    ax1.set_xlim(0, lim)
    fig.set_size_inches(4.5, 6)
    fig.subplots_adjust(right=0.97, left=0.4, bottom=0.15, top=0.9, wspace=0.4, hspace=0.2)
    plt.show()
    


def var_over_cylr_plots(
        df, 
        var,
        #hue,
        bin_col_0='time_bin', 
        bin_order_0=['0-60 s', '60-120 s', '120-300 s', '300-600 s'], 
        by_path=False
        ): 
    plt.rcParams['svg.fonttype'] = 'none'
    fig, ax = plt.subplots(1, 1) #len(bin_order_0)) 
    i = 0
    if by_path:
        df = df.groupby(['path', bin_col_0, 'cyl_r_bin'])[var].mean().reset_index()
    #df = smooth_vars(df, [var, ], t='cyl_r_bin', gb=['path', 'time_bin'], w=20)
    sns.lineplot(data=df, x='cyl_r_bin', y=var, ax=ax, hue=bin_col_0, hue_order=bin_order_0, palette='rocket')#hue=hue)
    sns.despine(ax=ax)
    fig.set_size_inches(3.5, 3)
    fig.subplots_adjust(right=0.97, left=0.17, bottom=0.15, top=0.9, wspace=0.4, hspace=0.4)
    plt.show()


def adjust_time(df):
    df['exp_type'] = df['path'].apply(classify_exp_type)
    for k, grp in df.groupby('exp_type'):
        if k == '10-20 min':
            idxs = grp.index.values
            df.loc[idxs, 'time (s)'] = df.loc[idxs, 'time (s)'] + 194 / 0.32
    return df


def timebin_box_plots(df, v0, v1, bin_col='time_bin'):
    fig, axs = plt.subplots(2, 1, sharex=True)
    ax0 = axs[0]
    ax1 = axs[1]
    plt.rcParams['svg.fonttype'] = 'none'
    print(f'getting {v0} means')
    data0 = df.groupby(['path', 'particle', 'psel', bin_col])[v0].mean().reset_index()
    print(f'getting {v1} means')
    data1 = df.groupby(['path', 'particle', 'psel', bin_col])[v1].mean().reset_index()
    print('preparing plots')
    sns.boxplot(data=data0, x=bin_col, y=v0, ax=ax0, hue='psel', palette='rocket')
    sns.despine(ax=ax0)
    sns.boxplot(data=data1, x=bin_col, y=v1, ax=ax1, hue='psel', palette='rocket')
    sns.despine(ax=ax1)
    fig.set_size_inches(4, 5)
    fig.subplots_adjust(right=0.97, left=0.17, bottom=0.13, top=0.95, wspace=0.4, hspace=0.2)
    plt.show()
    

def timebin_hist_plots(df, v0, bin_col='time_bin', cat='0-30 s'):
    fig, axs = plt.subplots(2, 1, sharex=True)
    ax0 = axs[0]
    ax1 = axs[1]
    plt.rcParams['svg.fonttype'] = 'none'
    print(f'getting {v0} means')
    #df = df[df[bin_col] == cat]
    data = df.groupby(['path', 'particle', 'psel'])[v0].mean().reset_index()
    data0 = data[data['psel'] == False]
    data1 = data[data['psel'] == True]
    print('preparing plots')
    sns.histplot(data=data0, x=v0,  ax=ax0, hue='psel', palette='rocket', bins=18)
    sns.despine(ax=ax0)
    sns.histplot(data=data1, x=v0, ax=ax1, hue='psel', palette='rocket', bins=14)
    sns.despine(ax=ax1)
    fig.set_size_inches(4, 5)
    fig.subplots_adjust(right=0.97, left=0.17, bottom=0.13, top=0.95, wspace=0.4, hspace=0.2)
    plt.show()


def count_over_cylr_plots(
        df, 
        #hue,
        bin_col_0='time_bin', 
        bin_order_0=['0-120 s', '120-300 s', '300-600 s'], 
        ):
    plt.rcParams['svg.fonttype'] = 'none'
    fig, ax = plt.subplots(1, 1) #len(bin_order_0)) 
    i = 0
    data = df.groupby(['path', 'cyl_r_bin', 'time_bin']).apply(count).reset_index().rename(columns={0 : 'count'})
    sns.lineplot(data=data, x='cyl_r_bin', y='count', ax=ax, hue=bin_col_0, hue_order=bin_order_0, palette='rocket')#hue=hue)
    sns.despine(ax=ax)
    fig.set_size_inches(3.5, 3)
    fig.subplots_adjust(right=0.97, left=0.17, bottom=0.15, top=0.9, wspace=0.4, hspace=0.4)
    plt.show()


def count(grp):
    return len(pd.unique(grp['particle']))


def com_estimates_plus_var(df, denscol, vcol, save_path):
    data = df.groupby(['path', 'cyl_r_bin', 'time_bin'])[denscol].mean()
    data0 = df.groupby(['path', 'cyl_r_bin', 'time_bin'])[vcol].mean()
    data = pd.concat([data, data0], axis=1).reset_index()
    data = smooth_vars(data, [denscol, vcol], t='cyl_r_bin', gb=['path', 'time_bin'], w=5)
    res = defaultdict(list)
    data = data.sort_values('cyl_r_bin')
    for k, grp in data.groupby(['path', 'time_bin']):
        res['path'].append(k[0])
        res['time_bin'].append(k[1])
        cr = grp['cyl_r_bin'].values
        d = grp[denscol].values
        v = grp[vcol].values
        di = np.argmax(d)
        dm = d[di]
        res['density max'].append(dm)
        dcr = cr[di]
        res['density peak (um)'].append(dcr)
        vi = np.argmax(v)
        vm = v[vi]
        res['variable max'].append(vm)
        vcr = cr[vi]
        res['variable peak (um)'].append(vcr)
        res['variable'].append(vcol)
    res = pd.DataFrame(res)
    res.to_csv(save_path)
    plt.rcParams['svg.fonttype'] = 'none'
    fig, axs = plt.subplots(1, 2)
    ax0 = axs[0]
    ax1 = axs[1]
    sns.scatterplot(data=res, x='density peak (um)', y='variable peak (um)', hue='time_bin', ax=ax0)
    sns.despine(ax=ax0)
    sns.scatterplot(data=res, x='density max', y='variable max', hue='time_bin', ax=ax1)
    sns.despine(ax=ax1)
    fig.set_size_inches(3.5, 3)
    fig.subplots_adjust(right=0.97, left=0.17, bottom=0.15, top=0.9, wspace=0.4, hspace=0.4)
    plt.show()


def com_vs_2_var(df0, df1, tbin_dens='0-60 s', tbin_var='300-600 s'):
    res = []
    for df in [df0, df1]:
        dens = df[df['time_bin'] == tbin_dens].drop(columns=['variable max', 'variable peak (um)', 'variable']).set_index(['path'])
        var = df[df['time_bin'] == tbin_var].drop(columns=['density max', 'density peak (um)']).set_index(['path'])
        out = pd.concat([dens, var], axis=1)
        res.append(out)
    res = pd.concat(res)
    fig, axs = plt.subplots(1, 2)
    ax0 = axs[0]
    ax1 = axs[1]
    plt.rcParams['svg.fonttype'] = 'none'
    sns.scatterplot(data=res, x='density peak (um)', y='variable peak (um)', hue='variable', ax=ax0, palette='rocket')
    sns.despine(ax=ax0)
    sns.scatterplot(data=res, x='density max', y='variable max', hue='variable', ax=ax1, palette='rocket')
    sns.despine(ax=ax1)
    fig.set_size_inches(3.5, 3)
    fig.subplots_adjust(right=0.97, left=0.17, bottom=0.15, top=0.9, wspace=0.4, hspace=0.4)
    plt.show()


def get_correct_times(df, tbin_dens='0-60 s', tbin_var='300-600 s'):
    dens = df[df['time_bin'] == tbin_dens].drop(columns=['variable max', 'variable peak (um)', 'variable']).set_index(['path'])
    var = df[df['time_bin'] == tbin_var].drop(columns=['density max', 'density peak (um)']).set_index(['path'])
    out = pd.concat([dens, var], axis=1)
    return out


def boxplot_for_peaks(data):
    data = get_correct_times(data)
    fig, axs = plt.subplots(1, 2)
    ax0 = axs[0]
    ax1 = axs[1]
    dens = data['density peak (um)'].values
    # stats
    print(f'n = {len(dens)}')
    print('density mean and SEM')
    print(dens.mean(), dens.std() / np.sqrt(len(dens)))
    print('density 95% CI')
    print(stats.t.interval(alpha=0.95, df=len(dens)-1, loc=np.mean(dens), scale=stats.sem(dens)) )
    var = data['variable peak (um)'].values
    print('variable mean and SEM')
    print(var.mean(), var.std() / np.sqrt(len(var)))
    print('variable 95% CI')
    print(stats.t.interval(alpha=0.95, df=len(var)-1, loc=np.mean(var), scale=stats.sem(var)) )
    plt.rcParams['svg.fonttype'] = 'none'
    sns.boxplot(data=data, x='variable', y='variable peak (um)', hue='variable', ax=ax0, palette='rocket')
    sns.stripplot(data=data, x='variable', y='variable peak (um)', hue='variable', 
                  ax=ax0, palette='rocket', dodge=True, edgecolor = 'white', linewidth=0.3, jitter=True, size=6)
    ax0.set_title('variable (300-600 s)')
    sns.despine(ax=ax0)
    sns.boxplot(data=data,x='variable', y='density peak (um)', hue='variable', ax=ax1, palette='rocket')
    sns.stripplot(data=data,x='variable', y='density peak (um)', hue='variable', 
                  ax=ax1, palette='rocket', dodge=True, edgecolor = 'white', linewidth=0.3, jitter=True, size=6)
    sns.despine(ax=ax1)
    fig.set_size_inches(2.5, 3)
    ax1.set_title('density (0-60 s)')
    fig.subplots_adjust(right=0.97, left=0.22, bottom=0.17, top=0.9, wspace=0.99, hspace=0.4)
    plt.show()


def barplot_for_peaks(data):
    data = get_correct_times(data)
    fig, axs = plt.subplots(1, 2)
    ax0 = axs[0]
    ax1 = axs[1]
    dens = data['density peak (um)'].values
    # stats
    print(f'n = {len(dens)}')
    print('density mean and SEM')
    print(dens.mean(), dens.std() / np.sqrt(len(dens)))
    print('density 95% CI')
    print(stats.t.interval(alpha=0.95, df=len(dens)-1, loc=np.mean(dens), scale=stats.sem(dens)) )
    var = data['variable peak (um)'].values
    print('variable mean and SEM')
    print(var.mean(), var.std() / np.sqrt(len(var)))
    print('variable 95% CI')
    print(stats.t.interval(alpha=0.95, df=len(var)-1, loc=np.mean(var), scale=stats.sem(var)) )
    plt.rcParams['svg.fonttype'] = 'none'
    sns.barplot(data=data, x='variable', y='variable peak (um)', hue='variable', ax=ax0, palette='rocket', capsize=0.3)
    sns.stripplot(data=data, x='variable', y='variable peak (um)', hue='variable', 
                  ax=ax0, palette='rocket', dodge=True, edgecolor = 'white', linewidth=0.3, jitter=True, size=6)
    ax0.set_title('variable (300-600 s)')
    sns.despine(ax=ax0)
    sns.barplot(data=data,x='variable', y='density peak (um)', hue='variable', ax=ax1, palette='rocket', capsize=0.3)
    sns.stripplot(data=data,x='variable', y='density peak (um)', hue='variable', 
                  ax=ax1, palette='rocket', dodge=True, edgecolor = 'white', linewidth=0.3, jitter=True, size=6)
    sns.despine(ax=ax1)
    fig.set_size_inches(2.5, 3)
    ax1.set_title('density (0-60 s)')
    fig.subplots_adjust(right=0.97, left=0.22, bottom=0.17, top=0.9, wspace=0.99, hspace=0.4)
    plt.show()


def inside_injury_correlation_platelet(
        df0, 
        df1, 
        x='nb_density_15', 
        y0='fibrin average intensity', 
        y1='p-sel average intensity', 
        hue_var='cyl_radial', 
        region='center'
        ):
    # first add region category and 
    df0 = df0[df0['time (s)'] > 300]
    df1 = df1[df1['time (s)'] > 300]
    if region is not None:
        df0 = df0[df0['region'] == 'center']
        df1 = df1[df1['region'] == 'center']
    data0 = df0.groupby(['path', 'particle'])[[x, y0, hue_var]].mean().reset_index()
    data1 = df1.groupby(['path', 'particle'])[[x, y1, hue_var]].mean().reset_index()
    
    # stats 
    print(y0)
    xvar = data0[x].values
    yvar = data0[y0].values
    res = stats.linregress(xvar, yvar)
    print(res)
    print(y1)
    xvar = data1[x].values
    yvar = data1[y1].values
    res = stats.linregress(xvar, yvar)
    print(res)

    # plots
    fig, axs = plt.subplots(1, 2)
    sns.scatterplot(data=data0, x=x, y=y0, palette='rocket', ax=axs[0], hue=hue_var, alpha=0.4)
    sns.scatterplot(data=data1, x=x, y=y1, palette='rocket', ax=axs[1], hue=hue_var, alpha=0.4)
    sns.despine(ax=axs[0])
    fig.set_size_inches(6, 3)
    fig.subplots_adjust(right=0.97, left=0.22, bottom=0.17, top=0.95, wspace=0.4, hspace=0.4)
    plt.show()




def inside_injury_correlation_path_cylrbin(
        df0, 
        df1, 
        x='nb_density_15', 
        y0='fibrin average intensity', 
        y1='p-sel average intensity', 
        bincol='cyl_r_bin',
        region='center'
        ):
    # first add region category and 
    df0 = df0[df0['time (s)'] > 300]
    df1 = df1[df1['time (s)'] > 300]
    if region is not None:
        df0 = df0[df0['region'] == 'center']
        df1 = df1[df1['region'] == 'center']
    data0 = df0.groupby(['path', 'particle', bincol])[[x, y0]].mean().reset_index()
    data1 = df1.groupby(['path', 'particle', bincol])[[x, y1]].mean().reset_index()
    data0 = data0.groupby(['path', bincol])[[x, y0]].mean().reset_index()
    data1 = data1.groupby(['path', bincol])[[x, y1]].mean().reset_index()
    # stats 
    print(y0)
    xvar = data0[x].values
    yvar = data0[y0].values
    res = stats.linregress(xvar, yvar)
    print(res)
    print(y1)
    xvar = data1[x].values
    yvar = data1[y1].values
    res = stats.linregress(xvar, yvar)
    print(res)

    # plots
    fig, axs = plt.subplots(1, 2)
    sns.scatterplot(data=data0, x=x, y=y0, palette='rocket', ax=axs[0], hue=bincol, alpha=0.6)
    sns.scatterplot(data=data1, x=x, y=y1, palette='rocket', ax=axs[1], hue=bincol, alpha=0.6)
    sns.despine(ax=axs[0])
    sns.despine(ax=axs[1])
    fig.set_size_inches(6, 3)
    fig.subplots_adjust(right=0.97, left=0.15, bottom=0.17, top=0.95, wspace=0.42, hspace=0.4)
    plt.show()



def inside_injury_correlation_path(
        df0, 
        df1, 
        x='nb_density_15', 
        y0='fibrin average intensity', 
        y1='p-sel average intensity', 
        yname='fluorescence', 
        region='center'
        ):
    # first add region category and 
    df0 = df0[df0['time (s)'] > 300]
    df1 = df1[df1['time (s)'] > 300]
    if region is not None:
        df0 = df0[df0['region'] == 'center']
        df1 = df1[df1['region'] == 'center']
    data0 = df0.groupby(['path', 'particle'])[[x, y0]].mean().reset_index()
    data1 = df1.groupby(['path', 'particle'])[[x, y1]].mean().reset_index()
    data0 = data0.groupby(['path'])[[x, y0]].mean().reset_index()
    data1 = data1.groupby(['path'])[[x, y1]].mean().reset_index()
    # stats 
    print(y0)
    xvar0 = data0[x].values
    yvar0 = data0[y0].values
    res0 = stats.linregress(xvar0, yvar0)
    print(res0)
    print(y1)
    xvar1 = data1[x].values
    yvar1 = data1[y1].values
    res1 = stats.linregress(xvar1, yvar1)
    print(res1)
    data0 = data0.rename(columns={y0 : yname})
    data0['variable'] = y0
    data1 = data1.rename(columns={y1 : yname})
    data1['variable'] = y1
    #data = pd.concat([data0, data1]).reset_index(drop=True)
    # plots
    plt.rcParams['svg.fonttype'] = 'none'
    fig, axs = plt.subplots(1, 2)
    sns.scatterplot(data=data0, x=x, y=yname, palette='rocket', ax=axs[0], hue='variable', alpha=0.7)
    sns.scatterplot(data=data1, x=x, y=yname, palette='rocket', ax=axs[1], hue='variable', alpha=0.7)
    sns.despine(ax=axs[0])
    sns.despine(ax=axs[1])
    plot_line(res0, xvar0, axs[0])
    plot_line(res1, xvar1, axs[1])
    fig.set_size_inches(6, 3)
    fig.subplots_adjust(right=0.97, left=0.15, bottom=0.17, top=0.95, wspace=0.42, hspace=0.4)
    plt.show()



def psel_pcnt_over_cylr_plots(
        df, 
        #hue,
        bin_col_0='time_bin', 
        bin_order_0=['0-120 s', '120-300 s', '300-600 s', ]#'600-1200 s'], 
        ):
    data = df.groupby(['path', 'cyl_r_bin', bin_col_0]).apply(psel_pcnt).reset_index()
    data = data.rename(columns={0 :  'p-selectin positive (%)'})
    plt.rcParams['svg.fonttype'] = 'none'
    fig, ax = plt.subplots(1, 1) #len(bin_order_0)) 
    i = 0
    data = smooth_vars(data, ['p-selectin positive (%)', ], t='cyl_r_bin', gb=['path', bin_col_0], w=5)
    sns.lineplot(data=data, x='cyl_r_bin', y='p-selectin positive (%)', ax=ax, hue=bin_col_0, hue_order=bin_order_0, palette='rocket')#hue=hue)
    sns.despine(ax=ax)
    fig.set_size_inches(3.5, 3)
    fig.subplots_adjust(right=0.97, left=0.17, bottom=0.15, top=0.9, wspace=0.4, hspace=0.4)
    plt.show()


def psel_pcnt(grp):
    n = len(pd.unique(grp['particle']))
    psel = grp[grp['psel'] == True]
    p = len(pd.unique(psel['particle']))
    return p / n * 100


def psel_pcnt_over_cylr_peaks_box(
        df, 
        #hue,
        bin_col_0='time_bin', 
        bin_order_0=['0-120 s', '120-300 s', '300-600 s','600-1200 s'], 
        ):
    data = df.groupby(['path', 'cyl_r_bin', bin_col_0]).apply(psel_pcnt).reset_index()
    data = data.rename(columns={0 :  'p-selectin positive (%)'})
    # get peaks
    #data = smooth_vars(data, ['p-selectin positive (%)', ], t='cyl_r_bin', gb=['path', bin_col_0], w=5)
    data = data.sort_values('cyl_r_bin')
    data = data.groupby(['path', bin_col_0]).apply(max_distance).reset_index()
    # PLOT
    plt.rcParams['svg.fonttype'] = 'none'
    fig, axs = plt.subplots(1, 2) #len(bin_order_0)) 
    sns.boxplot(data=data, x=bin_col_0, y='distance from centre (um)', ax=axs[0], hue_order=bin_order_0, palette='rocket')#hue=hue)
    sns.stripplot(data=data, x=bin_col_0, y='distance from centre (um)', ax=axs[0], hue=bin_col_0, hue_order=bin_order_0, 
                  palette='rocket', edgecolor = 'white', linewidth=0.3, jitter=True, size=5)
    sns.despine(ax=axs[0])
    sns.boxplot(data=data, x=bin_col_0, y='p-selectin positive (%)', ax=axs[1], hue_order=bin_order_0, palette='rocket')#hue=hue)
    sns.stripplot(data=data, x=bin_col_0, y='p-selectin positive (%)', ax=axs[1], hue=bin_col_0, hue_order=bin_order_0, 
                  palette='rocket', edgecolor = 'white', linewidth=0.3, jitter=True, size=5,)
    sns.despine(ax=axs[0])
    sns.despine(ax=axs[1])
    fig.set_size_inches(4.5, 3)
    fig.subplots_adjust(right=0.97, left=0.17, bottom=0.15, top=0.9, wspace=0.4, hspace=0.4)
    plt.show()


def max_distance(grp):
    vals = grp['p-selectin positive (%)'].values
    max_idx = np.argmax(vals)
    max_val = vals[max_idx]
    max_cyl = grp['cyl_r_bin'].values[max_idx]
    out = pd.DataFrame({
        'p-selectin positive (%)' : [max_val, ], 
        'distance from centre (um)' : [max_cyl, ]
    })
    return out


def psel_pcnt_peaks_donutness_scatter(
        df, 
        ddf,
        #hue,
        bin_col_0='time_bin', 
        bin_choice='300-600 s', 
        donut_var="donutness magnetude",
        other_var="donutness duration (s)"#"mean platelet count"
        ):
    data = df[df[bin_col_0] == bin_choice]
    data = data.groupby(['path', 'cyl_r_bin']).apply(psel_pcnt).reset_index()
    data = data.rename(columns={0 :  'p-selectin positive (%)'})
    # get peaks
    data = data.sort_values('cyl_r_bin')
    data = data.groupby('path').apply(max_distance).reset_index()
    data = data.set_index('path')
    # donutness
    ddf = ddf.set_index('path')
    ddf = ddf[[donut_var, other_var, "mean platelet count"]]
    # concat
    data = pd.concat([data, ddf], axis=1).reset_index()
    data = data.dropna(subset=[donut_var, other_var, 'p-selectin positive (%)', "mean platelet count"])
    # STATS
    donut = data[donut_var].values
    count = data[other_var].values
    pselpcnt = data['p-selectin positive (%)'].values
    res0 = stats.linregress(count, pselpcnt)
    res0_1 = stats.pearsonr(count, pselpcnt, )
    print(f'Psel pcnt x {other_var}\n', res0, '\n', res0_1)
    res1 = stats.linregress(donut, pselpcnt)
    res1_1 = stats.pearsonr(count, pselpcnt)
    print(f'Psel pcnt x {donut_var}', res1, '\n', res1_1)
    # PLOT
    plt.rcParams['svg.fonttype'] = 'none'
    fig, axs = plt.subplots(1, 2, sharey=True) #len(bin_order_0)) 
    sns.scatterplot(data=data, x=other_var, y='p-selectin positive (%)', palette='rocket', 
                    hue="mean platelet count", ax=axs[0],alpha=0.9)
    sns.scatterplot(data=data, x=donut_var, y='p-selectin positive (%)', palette='rocket', 
                    hue="mean platelet count", ax=axs[1],alpha=0.9)
    sns.despine(ax=axs[0])
    sns.despine(ax=axs[1])
    plot_line(res0, count, axs[0])
    plot_line(res1, donut, axs[1])
    fig.set_size_inches(5, 3)
    fig.subplots_adjust(right=0.97, left=0.17, bottom=0.15, top=0.9, wspace=0.35, hspace=0.4)
    plt.show()


def plot_line(res, xvar, ax):
    xmin = 0
    xstd = (xvar.max() - xmin) / 100000 #xvar.std()
    xnext = xmin + xstd
    m = res[0]
    b = res[1]
    ymin = xmin * m + b
    ynext =  xnext * m + b
    ax.axline((xmin, ymin), (xnext, ynext), color="grey", linestyle="--")


def classify_exp_type(path):
    if path.find('exp5') != -1:
        return '10-20 min'
    elif path.find('exp3') != -1:
        return '0-10 min'
    else:
        return 'other'

# ------------
# Execute Code
# ------------

options = ['psel_n_fib', 'par4_biva', 'donut']
do = options[0]
if do == 'psel_n_fib':
    # Read data
    # ---------
    p1 = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/230919_p-selectin.parquet'
    p0 = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/211206_saline_df_spherical-coords.parquet'
    #p0 = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/211206_ctrl_df.parquet'
    df0 = pd.read_parquet(p0)
    df1 = pd.read_parquet(p1)
    
    # Adjust variables
    # ----------------
    #df['treatment'] = df['path'].apply(get_treatment_name)
    df0['treatment'] = df0['path'].apply(get_treatment_name)
    df1['treatment'] = df1['path'].apply(get_treatment_name)
    #df = pd.concat([df, df0]).reset_index(drop=True)
    print(df0.columns.values)
    #df = df[df['nrtracks'] > 1]
    df0 = df0[df0['nrtracks'] > 1]
    df1 = df1[df1['nrtracks'] > 1]
    df0 = time_seconds(df0)
    df1 = time_seconds(df1)
    df1 = df1[df1['treatment'] != 'cangrelor']
    df1 = df1[df1['treatment'] != 'DMSO (MIPS)']
    df1 = df1[df1['path'] != '220602_IVMTR145_Inj1_ctrl_exp3']

    # for saline / other
    # ------------------
    df0 = df0.rename(columns={'c1_mean' : 'fibrin average intensity'})

    # for psel
    # --------
    df1 = df1.rename(columns={'GaAsP Alexa 568: mean_intensity' : 'p-sel average intensity'})
    df1 = df1.rename(columns={'elongation' : 'elong'})
    df1 = adjust_time(df1)
    df1 = psel_bin(df1)
    df1['exp_type'] = df1['path'].apply(classify_exp_type)
    df1 = df1[df1['exp_type'] == '0-10 min']
    df1 = df1[df1['treatment'] != 'cangrelor']
    df1 = df1[df1['treatment'] != 'DMSO (MIPS)']
    df1 = df1[df1['path'] != '220602_IVMTR145_Inj1_ctrl_exp3']

    # Filtering and binning
    # ---------------------
    a = get_rs(a=300, lim=2)
    lbs_cyl = a[0:-1] #np.linspace(0, 100)[0:-1]#a[0:-1]
    ubs_cyl =  a[1:] #np.linspace(0, 100)[1:]#a[1:]
    df0 = add_binns(df0, lbs_cyl=lbs_cyl, ubs_cyl=ubs_cyl)
    df1 = add_binns(df1, lbs_cyl=lbs_cyl, ubs_cyl=ubs_cyl)

    # Density x fluor corr plots
    # --------------------------
    # FOR FIGURE 2
    #df1 = spherical_coordinates(df1)
    #df0 = add_region_category(df0)
    #df1 = add_region_category(df1)
    #inside_injury_correlation_path_cylrbin(df0, df1, region=None)
    #inside_injury_correlation_platelet(df0, df1, region=None)
    #inside_injury_correlation_path(df0, df1, region=None) # fig 2
    #df0 = df0[df0['zs'] < 10]
    #df1 = df1[df1['zs'] < 10]
    #inside_injury_correlation_platelet(df0, df1)
    #inside_injury_correlation_path(df0, df1) # fig 2

    # psel percent
    # ------------
    psel_pcnt_over_cylr_plots(df1)
    psel_pcnt_over_cylr_peaks_box(df1)





elif do == 'par4_biva':
    # Read data
    # ---------
    #p = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/211206_biva_df_cleaned.parquet'
    p = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/211206_par4--_df_cleaned.parquet'
    df = pd.read_parquet(p)
    print(df.columns.values)
    
    # Adjust variables
    # ----------------
    df = df.rename(columns={'c1_mean' : 'fibrin average intensity'})
    df = time_seconds(df)
    df['treatment'] = df['path'].apply(get_treatment_name)

    # Filtering and binning
    # ---------------------
    lbs = [0, 60, 120, 300, 600]
    ubs = [60, 120, 300, 600, 1200]
    a = get_rs(a=300, lim=2)
    lbs_cyl = np.linspace(0, 100)[0:-1]#a[0:-1]
    ubs_cyl =  np.linspace(0, 100)[1:]#a[1:]
    df = add_binns(df, lbs, ubs, lbs_cyl, ubs_cyl)
    #df = local_densification(df)

    # Dist from centre plots
    # ----------------------
    #var_over_cylr_plots(df, 'nb_density_15', by_path=True)
    #var_over_cylr_plots(df, 'fibrin average intensity')

    # Peaks plots/stats
    # -----------------
    save_path_0 = '/Users/abigailmcgovern/Data/platelet-analysis/toroidal_paper/data/231027_biva_COM_vs_fibrin.csv'
    #save_path_0 = '/Users/abigailmcgovern/Data/platelet-analysis/toroidal_paper/data/231027_par4--_COM_vs_fibrin.csv'
    com_estimates_plus_var(df, 'nb_density_15', 'fibrin average intensity', save_path_0)
    fib = pd.read_csv(save_path_0)
    #boxplot_for_peaks(fib) 
    barplot_for_peaks(fib)

elif do == 'donut':
    # Read data
    # ---------
    p1 = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/230919_p-selectin.parquet'
    p0 = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/211206_saline_df_spherical-coords.parquet'
    #p0 = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/211206_ctrl_df.parquet'
    df0 = pd.read_parquet(p0)
    df1 = pd.read_parquet(p1)

    
    # Adjust variables
    # ----------------
    #df['treatment'] = df['path'].apply(get_treatment_name)
    df0['treatment'] = df0['path'].apply(get_treatment_name)
    df1['treatment'] = df1['path'].apply(get_treatment_name)
    #df = pd.concat([df, df0]).reset_index(drop=True)
    print(df0.columns.values)
    #df = df[df['nrtracks'] > 1]
    df0 = df0[df0['nrtracks'] > 1]
    df1 = df1[df1['nrtracks'] > 1]
    df0 = time_seconds(df0)
    df1 = time_seconds(df1)

    # for saline / other
    # ------------------
    df0 = df0.rename(columns={'c1_mean' : 'fibrin average intensity'})

    # for psel
    # --------
    df1 = df1.rename(columns={'GaAsP Alexa 568: mean_intensity' : 'p-sel average intensity'})
    df1 = df1.rename(columns={'elongation' : 'elong'})
    df1 = adjust_time(df1)
    df1 = psel_bin(df1)
    df1['exp_type'] = df1['path'].apply(classify_exp_type)
    #df1 = df1[df1['exp_type'] == '0-10 min']
    df1 = df1[df1['treatment'] != 'cangrelor']
    df1 = df1[df1['treatment'] != 'DMSO (MIPS)']
    df1 = df1[df1['path'] != '220602_IVMTR145_Inj1_ctrl_exp3']
    
    # Filtering and binning
    # ---------------------
    df0 = add_binns(df0)
    df1 = add_binns(df1)

    # donutness
    # ---------
    dp = '/Users/abigailmcgovern/Data/platelet-analysis/toroidal_paper/data/p-selectin_summary_data_gt1tk.csv'
    ddf = pd.read_csv(dp)
    psel_pcnt_peaks_donutness_scatter(df1, ddf, donut_var='donutness magnetude')


# -------------
# Stats Results
# -------------

# fibrin average intensity - gt300 centre (tracked)
#LinregressResult(slope=-4660.615536386265, intercept=211.7967832092556, 
# rvalue=-0.02729479873759079, pvalue=0.001405294254710491, stderr=1459.0801363440305, intercept_stderr=3.5576936183794174)
#p-sel average intensity - gt300 centre (tracked)
#LinregressResult(slope=-38438.36439124316, intercept=333.3189305531607, 
# rvalue=-0.11847748612735147, pvalue=6.4280725808534835e-49, stderr=2605.543112642481, intercept_stderr=4.2898213727510965)

# fibrin average intensity - gt300 centre (tracked > 1 min)
#LinregressResult(slope=-12678.90249508302, intercept=237.21453691356416, rvalue=-0.06629592528216094, 
# pvalue=5.283557382893751e-12, stderr=1836.0573438678402, intercept_stderr=4.553776268777467)
#p-sel average intensity - gt300 centre (tracked > 1 min)
#LinregressResult(slope=-76977.94503111017, intercept=411.0965186353847, rvalue=-0.22679489143256445, 
# pvalue=1.4052631830459115e-91, stderr=3742.753655501332, intercept_stderr=6.511835843551378)

# fibrin average intensity - gt300 centre (tracked, < 20 um in z)
#LinregressResult(slope=-61058.1033238526, intercept=413.5767335652172, rvalue=-0.22971885627135455, pvalue=1.8664990928799095e-67, stderr=3470.8207248341982, intercept_stderr=9.072375488820667)
#p-sel average intensity - gt300 centre (tracked, < 20 um in z)
#LinregressResult(slope=-180264.44813844498, intercept=703.4719113727228, rvalue=-0.36320489725344235, pvalue=3.124908448953614e-160, stderr=6448.08199985978, intercept_stderr=11.856662277292424)

# fibrin average intensity - gt300 centre (tracked > 1 min, < 20 um in z)
#LinregressResult(slope=-64705.24526788891, intercept=425.4900042387217, rvalue=-0.23422769517127465, 
# pvalue=1.3808520838197632e-61, stderr=3852.3929521485247, intercept_stderr=10.020805579014397)
#p-sel average intensity - gt300 centre (tracked > 1 min, < 20 um in z)
#LinregressResult(slope=-192871.72155451807, intercept=717.4472063324722, rvalue=-0.40329891937998574, 
# pvalue=2.331458705276126e-130, stderr=7589.255628221798, intercept_stderr=13.806548866113394)

#fibrin average intensity - gt300 centre (tracked > 1 min, < 10 um in z)
#LinregressResult(slope=-83801.19573510304, intercept=549.7274963183881, rvalue=-0.22141824874178556,
# pvalue=3.0659534938224407e-24, stderr=8145.666808077341, intercept_stderr=20.25708928521325)
#p-sel average intensity - gt300 centre (tracked > 1 min, < 10 um in z)
#LinregressResult(slope=-283541.12887089135, intercept=908.3173777476807, rvalue=-0.5002782283150781, 
# pvalue=2.8073824444605927e-85, stderr=13456.393527476042, intercept_stderr=22.562549237106843)

#fibrin average intensity - cylbins x path gt300 centre (tracked)
#LinregressResult(slope=-20501.080756410414, intercept=235.4668346526939, rvalue=-0.35461816256830675, 
# pvalue=7.708324959649813e-10, stderr=3218.904142165182, intercept_stderr=7.323951232297491)
#p-sel average intensity
#LinregressResult(slope=-60009.86311515799, intercept=342.1462909220244, rvalue=-0.27655929081997926, 
# pvalue=4.2381151069598374e-08, stderr=10725.324738205574, intercept_stderr=14.110068127770862)

#fibrin average intensity - cylbins x path gt300 centre (tracked, z < 10)
#LinregressResult(slope=-79887.6575048971, intercept=524.7952074856984, rvalue=-0.39675312374359367, 
# pvalue=2.0090708481998786e-11, stderr=11396.954758875938, intercept_stderr=27.627739293244137)
#p-sel average intensity - cylbins x path gt300 centre (tracked > 1 min)
#LinregressResult(slope=-293431.1134920848, intercept=891.0348572577857, rvalue=-0.520069617060178, 
# pvalue=2.2758191657819552e-25, stderr=25982.82207145566, intercept_stderr=38.57759485532614)

# fibrin average intensity - path gt300 centre (tracked, z < 10)
#LinregressResult(slope=-2497.2608040761975, intercept=349.62630404760984, rvalue=-0.023120827277958587, 
# pvalue=0.934815260023657, stderr=29948.33918746127, intercept_stderr=71.46758322739497)
#p-sel average intensity - path gt300 centre (tracked)
#LinregressResult(slope=-274629.9515881951, intercept=892.5809648745267, rvalue=-0.678637606381906, 
# pvalue=0.0010034297011598167, stderr=70056.70812578246, intercept_stderr=100.18926666345156)


# fibrin average intensity - path gt300 all regions (tracked)
#LinregressResult(slope=4148.595235358677, intercept=173.81940238714017, rvalue=0.11617513430208153,
#  pvalue=0.005364448617788807, stderr=1484.291178981876, intercept_stderr=2.803365795162053)
#p-sel average intensity
#LinregressResult(slope=24381.53009819799, intercept=217.193749217265, rvalue=0.13237434497364545, 
# pvalue=3.224682338268504e-05, stderr=5837.7959214129305, intercept_stderr=5.522813619774087)
#fibrin average intensity
#LinregressResult(slope=7712.023187969742, intercept=183.29706179890275, rvalue=0.08083546485631495, 
# pvalue=0.7745853633484074, stderr=26373.704342930287, intercept_stderr=54.586708557239426)
#p-sel average intensity
#LinregressResult(slope=-43270.676300003026, intercept=294.4332757888368, rvalue=-0.16535245563406545,
#  pvalue=0.4860010399391377, stderr=60831.28482723351, intercept_stderr=53.66023366233684)


#fibrin average intensity - path (tracked)
#LinregressResult(slope=7712.023187969742, intercept=183.29706179890275, rvalue=0.08083546485631495, 
# pvalue=0.7745853633484074, stderr=26373.704342930287, intercept_stderr=54.586708557239426)
#p-sel average intensity
#LinregressResult(slope=-43270.676300003026, intercept=294.4332757888368, rvalue=-0.16535245563406545, 
# pvalue=0.4860010399391377, stderr=60831.28482723351, intercept_stderr=53.66023366233684)

#fibrin average intensity - path (tracked, bottom, centre)
#LinregressResult(slope=-2497.2608040761975, intercept=349.62630404760984, rvalue=-0.023120827277958587, 
# pvalue=0.934815260023657, stderr=29948.33918746127, intercept_stderr=71.46758322739497)
#p-sel average intensity
#LinregressResult(slope=-274629.9515881951, intercept=892.5809648745267, rvalue=-0.678637606381906, 
# pvalue=0.0010034297011598167, stderr=70056.70812578246, intercept_stderr=100.18926666345156)

#Psel pcnt x donutness duration (s)
# LinregressResult(slope=0.031144653629342576, intercept=3.906430522243827, rvalue=0.7625363725679793, pvalue=0.010326073008924816, stderr=0.009342122130308664, intercept_stderr=3.0538661445629756) 
# PearsonRResult(statistic=0.7625363725679793, pvalue=0.010326073008924819)
#Psel pcnt x donutness magnetude LinregressResult(slope=8.690491399643115, intercept=-4.069264371133032, rvalue=0.8084685313309895, pvalue=0.004639447002708278, stderr=2.2367207563177014, intercept_stderr=4.4102203695811895) 
# PearsonRResult(statistic=0.7625363725679793, pvalue=0.010326073008924819)


#fibrin average intensity - ctrl tracked
#LinregressResult(slope=7712.023187969742, intercept=183.29706179890275, rvalue=0.08083546485631495, 
# pvalue=0.7745853633484074, stderr=26373.704342930287, intercept_stderr=54.586708557239426)
#p-sel average intensity - ctrl tracked 
#LinregressResult(slope=-130573.03734635208, intercept=355.15809641517217, 
# rvalue=-0.7778972306794097, pvalue=0.008062981988411974, stderr=37292.07949266213, intercept_stderr=36.32552321867218)

# --------
# OLD CODE
# --------


# Plots
# -----

#sns.scatterplot(data=df, x='p-sel average intensity', y='densification (/um3/sec)', hue='nrtracks', palette="rocket_r", alpha=0.3)
#sns.lineplot(data=df, x='tracknr', y='nb_density_15_pcntf', hue='psel')
#sns.despine()
#plt.show()

#initial_plots(df, 'nb_density_15_pcntf', nrlim=600)
#initial_plots(df, 'nb_density_15', nrlim=600)
#initial_plots(df, 'p-sel average intensity', nrlim=194)
#initial_plots(df, 'densification (/um3/sec)', nrlim=194)

#df = smooth_vars(df, ['p-sel average intensity', ], t='cyl_r_bin', gb=['' ])
#var_over_cylr_plots(df, 'p-sel average intensity')

#var_over_cylr_plots(df, 'nb_density_15')
#var_over_cylr_plots(df, 'fibrin average intensity')
#var_over_cylr_plots(df, 'nb_density_15', bin_order_0=['0-60 s', '60-120 s', '120-300 s', '300-600 s'])

#count_over_cylr_plots(df)

#initial_plots_all(df, 'nb_density_15_pcntf', 'nb_density_15')
#timebin_box_plots(df, 'nb_density_15', 'p-sel average intensity')

#timebin_hist_plots(df, 'nrtracks')


#plot_corrs(df, 'p-sel average intensity', 'nb_density_15')
#plot_corrs(df, 'p-sel average intensity', 'nrtracks')
#plot_corrs(df, 'p-sel average intensity', 'densification (/um3/sec)')

#sp_new = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/p-sel_cleaned_with_new.parquet'
#df.to_parquet(sp_new)


# Peaks plots/stats
# -----------------

#save_path_0 = '/Users/abigailmcgovern/Data/platelet-analysis/toroidal_paper/data/231023_COM_vs_fibrin.csv'
#save_path_0 = '/Users/abigailmcgovern/Data/platelet-analysis/toroidal_paper/data/231024_biva_COM_vs_fibrin.csv'
#save_path_0 = '/Users/abigailmcgovern/Data/platelet-analysis/toroidal_paper/data/231024_par4--_COM_vs_fibrin.csv'
#com_estimates_plus_var(df, 'nb_density_15', 'fibrin average intensity', save_path_0)


#save_path_1 = '/Users/abigailmcgovern/Data/platelet-analysis/toroidal_paper/data/231023_COM_vs_psel.csv'
#com_estimates_plus_var(df, 'nb_density_15', 'p-sel average intensity', save_path_1)

#psel = pd.read_csv(save_path_1)
#fib = pd.read_csv(save_path_0)

#com_vs_2_var(fib, psel)

#boxplot_for_peaks(fib)
#boxplot_for_peaks(psel)


# Density correlations
# --------------------
#df1 = spherical_coordinates(df1)
#df0 = add_region_category(df0)
#df1 = add_region_category(df1)


# FOR FIGURE 2
#inside_injury_correlation_path_cylrbin(df0, df1, region=None)
#inside_injury_correlation_platelet(df0, df1, region=None)
#inside_injury_correlation_path(df0, df1, region=None) # fig 2
#df0 = df0[df0['zs'] < 10]
#df1 = df1[df1['zs'] < 10]
#inside_injury_correlation_platelet(df0, df1)
#inside_injury_correlation_path(df0, df1) # fig 2


#inside_injury_correlation_path(df0, df1, x='stab')
#inside_injury_correlation_path(df0, df1, x='dv')

#psel_pcnt_over_cylr_plots(df1)
#psel_pcnt_over_cylr_peaks_box(df1)
