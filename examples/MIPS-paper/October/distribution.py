import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from toolz import curry
from plateletanalysis.variables.basic import add_region_category, tracking_time_var, add_quadrant
from scipy import stats
import numpy as np


def show_hists(df, var, trlim, tlim, ls=False, xlim=None):
    plt.rcParams['svg.fonttype'] = 'none'
    df = df[df['nrtracks'] > trlim]
    df = df[df['time (s)'] > tlim]
    data = df.groupby(['treatment', 'path', 'particle'])[var].mean().reset_index()
    data = pd.DataFrame(data)
    fig, axs = plt.subplots(1, 2)
    df.groupby('particle')
    sns.histplot(data, x=var, hue='treatment', fill=True, ax=axs[0], 
                 stat="density", log_scale=ls, hue_order=['MIPS', 'DMSO (MIPS)'], palette="rocket")
    if xlim is not None:
        axs[0].set_xlim(xlim)
    sns.histplot(data, x=var, hue='treatment', fill=False, ax=axs[1], cumulative=True, 
                 element="step", common_norm=False, stat="density", log_scale=ls, 
                 hue_order=['MIPS', 'DMSO (MIPS)'], palette="rocket")
    if xlim is not None:
        axs[1].set_xlim(xlim)
    sns.despine(ax=axs[0])
    sns.despine(ax=axs[1])
    res = stats.ks_2samp(data[data['treatment'] == 'MIPS'][var].values, 
                       data[data['treatment'] == 'DMSO (MIPS)'][var].values)
    print(res)
    fig.set_size_inches(8, 3)
    fig.subplots_adjust(right=0.97, left=0.13, bottom=0.13, top=0.97, wspace=0.3, hspace=0.2)
    plt.show()


def show_hists_quads(df, var, trlim, tlim):
    df = df[df['nrtracks'] > trlim]
    df = df[df['time (s)'] > tlim]
    fig, axs = plt.subplots(3, 2)
    i = 0
    for k, sdf in df.groupby('region'):
        data = sdf.groupby(['treatment', 'path', 'particle'])[var].mean().reset_index()
        data = pd.DataFrame(data)
        df.groupby('particle')
        sns.histplot(data, x=var, hue='treatment', fill=False, ax=axs[i, 0], stat="density")
        sns.histplot(data, x=var, hue='treatment', fill=False, ax=axs[i, 1], cumulative=True, element="step", common_norm=False, stat="density")
        sns.despine(ax=axs[i, 0])
        sns.despine(ax=axs[i, 1])
        axs[i, 0].set_title(k)
        axs[i, 1].set_title(k)
        i += 1
    plt.show()


def mean_and_cat(grp):
    if isinstance(grp.values[0], str):
        return grp.values[0]
    else:
        return grp.mean()
    

def percentile_analysis(df, var, trlim, tlim, ls=False, xlim=None):
    plt.rcParams['svg.fonttype'] = 'none'
    sns.set_context('paper')
    sns.set_style('ticks')
    df = df[df['nrtracks'] > trlim]
    df = df[df['time (s)'] > tlim]
    data = df.groupby(['treatment', 'path', 'particle'])[var].mean().reset_index()
    data = pd.DataFrame(data)
    mips = data[data['treatment'] == 'MIPS'][var].values
    dmso = data[data['treatment'] == 'DMSO (MIPS)'][var].values
    percent = np.arange(1, 101) 
    mips_score = stats.scoreatpercentile(mips, percent)
    dmso_score = stats.scoreatpercentile(dmso, percent)
    mips_inhib = mips_score / dmso_score * 100
    pdata = {
        'percentile' : percent, 
        f'{var} MIPS' : mips_score, 
        f'{var} vehicle' : dmso_score, 
        f'{var} MIPS inhibition (%)' : mips_inhib
    }
    pdata = pd.DataFrame(pdata)
    fig, axs = plt.subplots(1, 2)
    ax0 = axs[0]
    ax1 = axs[1]
    sns.lineplot(data=pdata, x='percentile', y=f'{var} MIPS inhibition (%)', ax=ax0)
    ax0.axline((0, 100), (1, 100), color='grey', alpha=0.4, linestyle="--")
    sns.despine(ax=ax0)
    sns.lineplot(data=pdata, x=f'{var} vehicle', y=f'{var} MIPS', ax=ax1)
    p0 = dmso_score.min()
    p1 = mips_score.min()
    add = mips_score[1] - mips_score[0]
    ax1.axline((p0, p0), (p0 + add, p0 + add), color='grey', alpha=0.4, linestyle="--")
    sns.despine(ax=ax1)
    fig.set_size_inches(6, 2.5)
    fig.subplots_adjust(right=0.97, left=0.12, bottom=0.17, top=0.97, wspace=0.35, hspace=0.2)
    plt.show()



def percentile_analysis_regions(df, var, trlim, tlim, ls=False, xlim=None):
    plt.rcParams['svg.fonttype'] = 'none'
    sns.set_context('paper')
    sns.set_style('ticks')
    df = df[df['nrtracks'] > trlim]
    df = df[df['time (s)'] > tlim]
    data = df.groupby(['treatment', 'path', 'particle', 'region'])[var].mean().reset_index()
    data = pd.DataFrame(data)
    i = 0
    pdata = defaultdict(list)
    for k, grp in data.groupby('region'):
        mips = grp[grp['treatment'] == 'MIPS'][var].values
        dmso = grp[grp['treatment'] == 'DMSO (MIPS)'][var].values
        percent = np.arange(1, 101) 
        mips_score = stats.scoreatpercentile(mips, percent)
        dmso_score = stats.scoreatpercentile(dmso, percent)
        mips_inhib = mips_score / dmso_score * 100
        r = [k, ] * len(percent)
        pdata = {
            'region' : np.concatenate([pdata['region'], r]),
            'percentile' : np.concatenate([pdata['percentile'], percent]), 
            f'{var} MIPS' : np.concatenate([pdata[f'{var} MIPS'], mips_score]), 
            f'{var} vehicle' : np.concatenate([pdata[f'{var} vehicle'], dmso_score]), 
            f'{var} MIPS inhibition (%)' : np.concatenate([pdata[f'{var} MIPS inhibition (%)'], mips_inhib])
        }
    pdata = pd.DataFrame(pdata)
    fig, axs = plt.subplots(1, 2)
    ax0 = axs[0]
    ax1 = axs[1]
    sns.lineplot(data=pdata, x='percentile', y=f'{var} MIPS inhibition (%)', ax=ax0, hue='region', palette="viridis")
    ax0.axline((0, 100), (1, 100), color='grey', alpha=0.4, linestyle="--")
    if ls:
        ax0.set_xscale('log')
    sns.despine(ax=ax0)
    sns.lineplot(data=pdata, x=f'{var} vehicle', y=f'{var} MIPS', ax=ax1, hue='region', palette="viridis")
    p0 = dmso_score.min()
    #p1 = mips_score.min()
    unique = pd.unique(pdata[f'{var} MIPS'])
    add = unique[1] - unique[0]
    ax1.axline((p0, p0), (p0 + add, p0 + add), color='grey', alpha=0.4, linestyle="--")
    if ls:
        ax1.set_xscale('log')
    sns.despine(ax=ax1)
    fig.set_size_inches(6, 2.5)
    fig.subplots_adjust(right=0.97, left=0.12, bottom=0.17, top=0.97, wspace=0.35, hspace=0.2)
    plt.show()


def count_in_vehicle_percentile_bin(df, var, trlim, tlim):
    plt.rcParams['svg.fonttype'] = 'none'
    sns.set_context('paper')
    sns.set_style('ticks')
    df = df[df['nrtracks'] > trlim]
    df = df[df['time (s)'] > tlim]
    data = df.groupby(['treatment', 'path', 'particle', 'region', 'nrtracks', 'dist_c'])[var].mean().reset_index()
    data = pd.DataFrame(data)
    pdf = []
    for k, grp in data.groupby('region'):
        dmso = grp[grp['treatment'] == 'DMSO (MIPS)'][var].values
        percent = np.arange(1, 101, 5) 
        dmso_score = stats.scoreatpercentile(dmso, percent)
        start_val = stats.scoreatpercentile(dmso, 0)
        binning_func = pcnt_bin(start_val, percent, dmso_score)
        grp['pcnt_bin'] = grp[var].apply(binning_func)
        count = info_count(var)
        pdata = grp.groupby(['treatment', 'pcnt_bin']).apply(count).reset_index()
        pdata['region'] = k
        pdf.append(pdata)
    pdf = pd.concat(pdf).reset_index()
    mdf = pdf[pdf['treatment'] == 'MIPS']
    ddf = pdf[pdf['treatment'] == 'DMSO (MIPS)']
    for k, grp in mdf.groupby(['pcnt_bin', 'region']):
        dgrp = ddf[(ddf['pcnt_bin']==k[0]) &(ddf['region']==k[1])]
        idx = grp.index.values
        m0 = grp['count'].values
        d0 = dgrp['count'].values
        mdf.loc[idx, 'count (% veh)'] = m0 / d0 * 100
        m1 = grp['nrtracks'].values
        d1 = dgrp['nrtracks'].values
        mdf.loc[idx, 'nrtracks (% veh)'] = m1 / d1 * 100
        m2 = grp['dist_c'].values
        d2 = dgrp['dist_c'].values
        mdf.loc[idx, 'dist_c (% veh)'] = m2 / d2 * 100
    fig, axs = plt.subplots(1, 4)
    sns.lineplot(data=ddf, x='pcnt_bin', y=var, hue='region', ax=axs[0], palette='viridis')
    sns.despine(ax=axs[0])
    sns.lineplot(data=mdf, x='pcnt_bin', y='count (% veh)', hue='region',ax=axs[1], palette='viridis')
    axs[1].axline((0, 100), (1, 100), color='grey', alpha=0.4, linestyle="--")
    sns.despine(ax=axs[1])
    sns.lineplot(data=mdf, x='pcnt_bin', y='nrtracks (% veh)', hue='region', ax=axs[2], palette='viridis')
    axs[2].axline((0, 100), (1, 100), color='grey', alpha=0.4, linestyle="--")
    sns.despine(ax=axs[2])
    sns.lineplot(data=mdf, x='pcnt_bin', y='dist_c (% veh)', hue='region', ax=axs[3], palette='viridis')
    axs[3].axline((0, 100), (1, 100), color='grey', alpha=0.4, linestyle="--")
    sns.despine(ax=axs[3])
    fig.set_size_inches(8, 2.5)
    fig.subplots_adjust(right=0.97, left=0.12, bottom=0.17, top=0.85, wspace=0.35, hspace=0.2)    
    plt.show()


@curry
def pcnt_bin(start_val, pcnt_bins, val_bins, val):
    sb = start_val
    for pb, ub in zip(pcnt_bins, val_bins):
        if val >= sb and val < ub:
            return pb
        
@curry
def info_count(var, grp):
    out = pd.DataFrame({
        'count' : [len(pd.unique(grp['particle'])), ], 
        'nrtracks' : [grp['nrtracks'].mean(), ], 
        'dist_c' :  [grp['dist_c'].mean(), ], 
        var :  [grp[var].mean(), ]
    })
    return out


def nrtracks_vs_initial_inhib(df, var, trlim, tlim,):
    plt.rcParams['svg.fonttype'] = 'none'
    sns.set_context('paper')
    sns.set_style('ticks')
    df = df[df['nrtracks'] > trlim]
    df = df[df['time (s)'] > tlim]
    df = df[df['tracknr'] < 4]
    data0 = df.groupby(['treatment', 'path', 'particle', 'region'])['nrtracks'].mean().reset_index()
    data1 = df.groupby(['treatment', 'path', 'particle', 'region'])[var].mean().reset_index().drop(columns=['treatment', 'path', 'particle', 'region'])
    data = pd.concat([data0, data1], axis=1).reset_index()
    pcnt_inhib_var = pcnt_inhib(var)
    inhib = data.groupby('nrtracks').apply(pcnt_inhib_var).reset_index()
    reg = pd.unique(data['region'])
    fig, axs = plt.subplots(1, len(reg), sharex=True, sharey=True)
    i = 0
    for k, grp in inhib.groupby('region'):
        ax = axs[i]
        sns.kdeplot(data=grp, x='nrtracks', y= f"{var} inhibition (%)", ax=ax, fill=True, cmap="rocket_r")
        sns.despine(ax=ax)
        ax.axline((0, 100), (1, 100), color='grey', alpha=0.4, linestyle="--")
        ax.set_title(k)
        i += 1
    fig.set_size_inches(8, 2.5)
    fig.subplots_adjust(right=0.97, left=0.12, bottom=0.17, top=0.85, wspace=0.35, hspace=0.2)    
    plt.show()


@curry
def pcnt_inhib(var, grp):
    grp = grp.reset_index()
    dmso = grp[grp['treatment'] == 'DMSO (MIPS)']
    veh_mean = dmso[var].mean()
    out = pd.DataFrame({
        'region' : grp[grp['treatment'] == 'MIPS']['region'], 
       # 'nrtracks' : grp[grp['treatment'] == 'MIPS']['nrtracks'], 
        f"{var} inhibition (%)" : grp[grp['treatment'] == 'MIPS'][var] / veh_mean * 100
    })
    return out



def smooth_vars(df, vars, w=20, t='time (s)', gb='path', add_suff=None):
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




def percentile_analysis_regions_trkbins(
        df, 
        var, 
        tlim, 
        tkbins = ((1, 10), (10, 30), (30, 600)), 
        category='quadrant'
        ):
    plt.rcParams['svg.fonttype'] = 'none'
    sns.set_context('paper')
    sns.set_style('ticks')
    df = df[df['time (s)'] > tlim]
    bin_func = bin_by_trk(tkbins)
    df['residency'] = df['nrtracks'].apply(bin_func)
    data = df.groupby(['treatment', 'path', 'particle', category, 'residency'])[var].mean().reset_index()
    data = pd.DataFrame(data)
    pdata = defaultdict(list)
    for k, grp in data.groupby([category, 'residency']):
        mips = grp[grp['treatment'] == 'MIPS'][var].values
        dmso = grp[grp['treatment'] == 'DMSO (MIPS)'][var].values
        percent = np.arange(1, 101) 
        mips_score = stats.scoreatpercentile(mips, percent)
        dmso_score = stats.scoreatpercentile(dmso, percent)
        mips_inhib = mips_score / dmso_score * 100
        r = [k[0], ] * len(percent)
        res = [k[1], ] * len(percent)
        pdata = {
            category : np.concatenate([pdata[category], r]),
            'residency' : np.concatenate([pdata['residency'], res]),
            'percentile' : np.concatenate([pdata['percentile'], percent]), 
            f'{var} MIPS' : np.concatenate([pdata[f'{var} MIPS'], mips_score]), 
            f'{var} vehicle' : np.concatenate([pdata[f'{var} vehicle'], dmso_score]), 
            f'{var} MIPS inhibition (%)' : np.concatenate([pdata[f'{var} MIPS inhibition (%)'], mips_inhib])
        }
    pdata = pd.DataFrame(pdata)
    fig, axs = plt.subplots(1, len(pd.unique(pdata[category])), sharey=True)
    i = 0
    for k, grp in pdata.groupby(category):
        ax = axs[i]
        sns.lineplot(data=grp, x='percentile', y=f'{var} MIPS inhibition (%)', ax=ax, hue='residency', palette="icefire")
        ax.axline((0, 100), (1, 100), color='grey', alpha=0.4, linestyle="--")
        sns.despine(ax=ax)
        ax.set_title(k)
        i += 1
    fig.set_size_inches(8, 2.5)
    fig.subplots_adjust(right=0.97, left=0.10, bottom=0.17, top=0.87, wspace=0.4, hspace=0.2)
    plt.show()


@curry
def bin_by_trk(bins, val):
    for l, u in bins:
        if val > l and val <= u:
            n = f"{l}-{u} s"
            return n


def tkbinned_boxplots(df, var, tlim, tkbins, category='quadrant'):
    plt.rcParams['svg.fonttype'] = 'none'
    sns.set_context('paper')
    sns.set_style('ticks')
    df = df[df['time (s)'] > tlim]
    bin_func = bin_by_trk(tkbins)
    df['residency'] = df['nrtracks'].apply(bin_func)
    data = df.groupby(['treatment', 'path', 'particle', category, 'residency'])[var].mean().reset_index()
    data = pd.DataFrame(data)
    edata = data.groupby(['treatment', 'path', category, 'residency'])[var].mean().reset_index()
    edata['platelet count'] = data.groupby(['treatment', 'path', category, 'residency'])[var].apply(count_rows).values
    #
    data = edata.reset_index()
    data = pd.DataFrame(data)
    fig, axs = plt.subplots(2, len(pd.unique(data[category])), sharex=True, sharey='row')
    i = 0
    for k, grp in data.groupby(category):
        ax0 = axs[0, i]
        sns.boxplot(data=grp, x='residency', y='platelet count', hue='treatment', ax=ax0)
        ax0.set_title(k)
        ax0.set_yscale('log')
        sns.despine(ax=ax0)
        ax1 = axs[1, i]
        sns.boxplot(data=grp, x='residency', y=var, hue='treatment', ax=ax1)
        sns.despine(ax=ax1)
        i += 1
    fig.set_size_inches(8, 5)
    fig.subplots_adjust(right=0.97, left=0.127, bottom=0.087, top=0.97, wspace=0.4, hspace=0.3)
    plt.show()

def count_rows(var):
    return len(var)


files = ['230301_MIPS_and_DMSO.parquet', '211206_veh-mips_df.parquet', '211206_mips_df.parquet']
d = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/'
df = [pd.read_parquet(os.path.join(d, f)) for f in files]
df = pd.concat(df).reset_index()
df = df[df['treatment'] != 'DMSO (salgav)']

#df = df[df['inside_injury']== False]
df = df[df['size'] == 'large']
df = tracking_time_var(df)
df = add_quadrant(df)

# sl outside KstestResult(statistic=0.10544659423525826, pvalue=1.7115690539877655e-192, 
#   statistic_location=0.002115108314751309, statistic_sign=1)
# large outside KstestResult(statistic=0.12896165010934835, pvalue=1.447331329237673e-198, 
#   statistic_location=0.002221343975891045, statistic_sign=1)
#show_hists_quads(df, 'nb_density_15', 10, 300)

#show_hists(df, 'nrtracks', 1, 300)
# KstestResult(statistic=0.07427888361979529, pvalue=0.0, statistic_location=2.0, statistic_sign=1)
#show_hists(df, 'nrtracks', 10, 300)
# KstestResult(statistic=0.07351916293756766, pvalue=1.1446716125562698e-64, statistic_location=32.0, statistic_sign=-1)
#show_hists(df, 'stab', 10, 300, xlim=[0, 4])
#KstestResult(statistic=0.07043887218276501, pvalue=2.1583156575187756e-59, statistic_location=1.521356146042228, statistic_sign=1)

#show_hists(df, 'dvy', 10, 300, xlim=(-2, 2))
# KstestResult(statistic=0.08836731547393273, pvalue=2.7825546300358883e-93, statistic_location=-0.15038418444562782, statistic_sign=1)
#show_hists_quads(df, 'dvy', 10, 300)

## new histograms
#show_hists(df, 'nb_density_15', 1, 300)
## KstestResult(statistic=0.30866248104348404, pvalue=0.0, statistic_location=0.0017006000400745112, statistic_sign=1)
#show_hists(df, 'nb_density_15', 10, 300)
## KstestResult(statistic=0.2018661328544722, pvalue=6.287573521027101e-251, statistic_location=0.002083397151348129, statistic_sign=1)
#show_hists(df, 'stab', 1, 300, xlim=[0, 4])
## KstestResult(statistic=0.1640136399773625, pvalue=0.0, statistic_location=4.278307123438953, statistic_sign=-1)
#show_hists(df, 'stab', 10, 300, xlim=[0, 4])
## KstestResult(statistic=0.09375298635959073, pvalue=4.410713585627508e-54, statistic_location=1.3182710355710914, statistic_sign=1)
#show_hists(df, 'dvy', 1, 300, xlim=(-2, 2))
## KstestResult(statistic=0.06846551359903347, pvalue=3.189188848068151e-149, statistic_location=-0.47418623862568765, statistic_sign=1)
#show_hists(df, 'dvy', 10, 300, xlim=(-2, 2))
## KstestResult(statistic=0.14012570105036531, pvalue=1.4610266494578635e-120, statistic_location=-0.22090394985154918, statistic_sign=1)
#show_hists(df, 'ca_corr', 1, 300)
## KstestResult(statistic=0.07123615456662563, pvalue=1.599401291485772e-161, statistic_location=1.7489048593538672, statistic_sign=-1)
#show_hists(df, 'ca_corr', 10, 300)
## KstestResult(statistic=0.09381002462930615, pvalue=3.793925653579071e-54, statistic_location=2.526481798545825, statistic_sign=-1)

#percentile_analysis(df, 'nb_density_15', 10, 300)
#percentile_analysis_regions(df, 'nb_density_15', 10, 300)
#percentile_analysis_regions(df, 'nb_density_15', 1, 300)
#percentile_analysis_regions(df, 'ca_corr', 1, 300)
#percentile_analysis_regions(df, 'ca_corr', 10, 300)
#percentile_analysis_regions(df, 'stab', 1, 300)
#percentile_analysis_regions(df, 'stab', 10, 300)

#nrtracks_vs_initial_inhib(df, 'nb_density_15', 1, 300)
#nrtracks_vs_initial_inhib(df, 'stab', 10, 300)
#nrtracks_vs_initial_inhib(df, 'ca_corr', 10, 300)

#count_in_vehicle_percentile_bin(df, 'nb_density_15', 1, 300)
#count_in_vehicle_percentile_bin(df, 'nb_density_15', 10, 300)

#count_in_vehicle_percentile_bin(df, 'stab', 1, 300)
#count_in_vehicle_percentile_bin(df, 'stab', 10, 300)

#count_in_vehicle_percentile_bin(df, 'ca_corr', 1, 300)
#count_in_vehicle_percentile_bin(df, 'ca_corr', 10, 300)


#percentile_analysis_regions_trkbins(df, 'nb_density_15', 300, tkbins=((1, 15), (15, 30), (30, 300), (300, 600)))
#percentile_analysis_regions_trkbins(df, 'stab', 300, tkbins=((1, 15), (15, 30), (30, 300), (300, 600)))
#percentile_analysis_regions_trkbins(df, 'dv', 300, tkbins=((1, 15), (15, 30), (30, 300), (300, 600)))

tkbinned_boxplots(df, 'nb_density_15', 300, tkbins=((1, 15), (15, 30), (30, 300), (300, 600)), category='quadrant')
tkbinned_boxplots(df, 'stab', 300, tkbins=((1, 15), (15, 30), (30, 300), (300, 600)), category='quadrant')
tkbinned_boxplots(df, 'dv', 300, tkbins=((1, 15), (15, 30), (30, 300), (300, 600)), category='quadrant')
