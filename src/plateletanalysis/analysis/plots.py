import pandas as pd
import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib
import numpy as np

# MAIN FUNCTIONS
# - abs_and_pcnt_timeplots (figure 2)
# - quantile_analysis_plots (figure 3)
# - inside_and_outside_injury_barplots (figure 3)
# - regions_abs_and_pcnt_timeplots (figures 4, 5)
# - individual_exp_inside_outside_timeplots (supplement)

# -----
# CMAPS
# -----

MIPS_order = ['DMSO (MIPS)', 'MIPS']
cang_order = ['saline','cangrelor']#['Saline','Cangrelor','Bivalirudin']
SQ_order = ['DMSO (SQ)', 'SQ']
pal_MIPS  = dict(zip(MIPS_order, sns.color_palette('Blues')[2::3]))
pal_cang = dict(zip(cang_order, sns.color_palette('Oranges')[2::3]))
pal_SQ = dict(zip(SQ_order, sns.color_palette('Greens')[2::3]))
pal1 = {**pal_MIPS,**pal_cang,**pal_SQ}


# ---------------------
# General summary plots
# ---------------------

def abs_and_pcnt_timeplots(
        df, 
        peaks,
        abs_tx='MIPS', 
        abs_ctl='DMSO (MIPS)',
        pcnt_treatments=('MIPS', 'SQ', 'cangrelor'), 
        pcnt_controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline'),
        names=('platelet count raw', 'platelet density (um^-3) raw'),
        pcnt_names=('platelet count raw pcnt', 'platelet density (um^-3) raw pcnt'),
        time_col='time (s)',
        hue='treatment', 
    ):

    '''
    This originally worked with data (df) from another function, but will also work
    with the output of stats.rolling_counts_and_growth_plus_peaks(). 

    df: pd.DataFrame
        Output 0 from stats.rolling_counts_and_growth_plus_peaks()
    peaks: pd.DataFrame
        Output 1 from stats.rolling_counts_and_growth_plus_peaks()
    pcnt_treatments: tuple of str
        Category names of the treatment conditions in the 'treatment' var. 
        As assignmed by basic.get_treatment_name. Thes will be plotted as
        pcnt of vehicle. 
    pcnt_controls: tuple of str
        Category names of the control conditions in the 'treatment' var. 
        As assignmed by basic.get_treatment_name
    abs_tx: str
        Name of the treatment for which to plot absolute data.
    abs_ctl: str
        Name of the control for which to plot absolute data.
    names: tuple of str
        Column names for the absolute variables to be plotted in the subplots on
        the left. 
    pcnt_names: tuple of str
        Column names for the pcnt of vehicle variables to be plotted in the subplots on
        the right. 
    '''
    sns.set_context('paper')
    sns.set_style('ticks')
    _add_time_to_peak(df, peaks, pcnt_treatments, pcnt_controls)
    df = df.sort_values(time_col)
    dfs = []
    for t in pcnt_treatments:
        sdf = df[df['treatment'] == t]
        dfs.append(sdf)
    txdf = pd.concat(dfs).reset_index(drop=True)
    if time_col == 'time (s)':
        max_t = txdf[time_col].max()
        txdf = txdf[txdf[time_col] < max_t - 4]
    fig, axs = plt.subplots(len(names), 2, sharex=False, sharey=False)
    mips = [df[df['treatment'] == abs_tx], df[df['treatment'] == abs_ctl]]
    mips = pd.concat(mips).reset_index(drop=True)
    if time_col == 'time (s)':
        max_t = mips[time_col].max()
        mips = mips[mips[time_col] < max_t - 4]
    for i, name in enumerate(names):
        _add_rolled(name, mips)
        name_pcnt = pcnt_names[i]
        _add_rolled(name_pcnt, txdf)
        p = df[name_pcnt].min()
        sns.lineplot(data=mips, x=time_col, y=name, hue=hue, ax=axs[i, 0], errorbar=("se", 1), palette=pal1)
        sns.lineplot(data=txdf, x='time from peak count (s)', y=name_pcnt, hue=hue, ax=axs[i, 1], errorbar=("se", 1), palette=pal1)
        axs[i, 1].axline((p, 100), (p + 1, 100), color='grey', alpha=0.5)
        axs[i, 1].axline((0, 99), (0, 100), color='grey', alpha=0.5, ls='--')
    fig.subplots_adjust(right=0.95, left=0.17, bottom=0.11, top=0.95, wspace=0.45, hspace=0.4)
    fig.set_size_inches(5, 4)
    sns.despine(fig)
    plt.show()


def _add_time_to_peak(df, peaks, treatments, controls):
    for i, ctl in enumerate(controls):
        tx = treatments[i]
        sdf = pd.concat([df[df['treatment'] == tx], df[df['treatment'] == ctl]])
        ctl_peaks = peaks[peaks['treatment'] == ctl]
        peak = ctl_peaks['time peak count'].mean()
        tfp = sdf['time (s)'] - peak
        idx = sdf.index.values
        df.loc[idx, 'time from peak count (s)'] = tfp


def _add_rolled(col, df):
    for k, grp in df.groupby('path'):
        idxs = grp.index.values
        roll = grp[col].rolling(window=8,center=False).mean()
        df.loc[idxs, col] = roll


# -----------------------
# Quantile analysis plots
# -----------------------

def quantile_analysis_plots(
        data, 
        variables, 
        save_path,
        treatments=('MIPS', 'SQ', 'cangrelor'), 
        controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline'), 
        insideout=False
        ):
    sns.set_context('paper')
    sns.set_style('ticks')
    # generate data
    dfs = []
    quantiles = {
        'quantile' : [], 
        'treatment' : [], 
        'tx_val' : [], 
        'variable' : [], 
    }
    if insideout:
        quantiles['inside injury'] = []
    for i, tx in enumerate(treatments):
        if insideout:
            for location in [True, False]:
                data_l = data[data['inside injury'] == location]
                df = get_percentile_data(data_l, tx, i, controls, variables, quantiles)
                dfs.append(df)
        else:
            df = get_percentile_data(data, tx, i, controls, variables, quantiles)
            dfs.append(df)
    df = pd.concat(dfs).reset_index(drop=True)
    df.to_csv(save_path)
    quantiles = pd.DataFrame(quantiles)
    n = Path(save_path).stem + '_quantiles.csv'
    sp_q = os.path.join(Path(save_path).parents[0], n)
    quantiles.to_csv(sp_q)
    # Make plots
    fig, ax = plt.subplots(len(variables), 2)
    for i, v in enumerate(variables):
        mqdf  = quantiles[(quantiles['treatment'] == 'MIPS') & (quantiles['variable'] == v)]
        dqdf  = quantiles[(quantiles['treatment'] == 'DMSO (MIPS)') & (quantiles['variable'] == v)]
        ax0 = ax[i, 0]
        ax0.axline((0, 100), (1, 100), color='grey', alpha=0.5)
        y0 = f'{v} pcnt veh'
        sns.lineplot(data=df, x='Percentile', y=y0, ax=ax0, hue='treatment', marker='o', palette=pal1)
        ax1 = ax[i, 1]
        x1 = f'Control {v}'
        y1 = f'Treatment {v}'
        p0 = df[x1].min()
        p1 = df[y1].min()
        # grey line with gradient of 1
        ax1.axline((p0, p0), (p0 + 1, p0 + 1), color='grey', alpha=0.5)
        # Blue line at MIPS == 50%
        M50 = mqdf[mqdf['quantile'] == 50]['tx_val'].values[0]
        ax1.axline((p0, M50), (p0 + 1, M50), color=pal1['MIPS'], alpha=0.4, ls='--')
        # Light blue line at DMSO == 50%
        D50 = dqdf[dqdf['quantile'] == 50]['tx_val'].values[0]
        ax1.axline((D50, p1), (D50, p1 + 1), color=pal1['DMSO (MIPS)'], alpha=0.4, ls='--')
        # plot it 
        sns.lineplot(data=df, x=x1, y=y1, ax=ax1, hue='treatment', marker='o', palette=pal1)
    matplotlib.rcParams.update({'font.size': 10})
    #fig.subplots_adjust(right=0.95, left=0.125, bottom=0.074, top=0.96, wspace=0.485, hspace=0.337)
    fig.subplots_adjust(right=0.95, left=0.17, bottom=0.11, top=0.95, wspace=0.45, hspace=0.4)
    fig.set_size_inches(6, 8)
    plt.show()


def get_percentile_data(data, tx, i, controls, variables, quantiles):
    txdf = data[data['treatment'] == tx]
    ctrldf = data[data['treatment'] == controls[i]]
    #df = pd.concat([txdf, ctrldf])
    if len(txdf) < len(ctrldf):
        n = len(txdf)
    else:
        n = len(ctrldf)
    pcnt = np.linspace(0, 100, n)
    df = {}
    df['Percentile'] = pcnt
    df['treatment'] = [tx, ] * len(pcnt)
    for v in variables:
        tx_score = [stats.scoreatpercentile(txdf[v].values, p) for p in pcnt]
        ctrl_score = [stats.scoreatpercentile(ctrldf[v].values, p) for p in pcnt]
        df[f'Control {v}'] = ctrl_score
        df[f'Treatment {v}'] = tx_score
        add_quantiles(quantiles, txdf, tx, v)
        add_quantiles(quantiles, ctrldf, controls[i], v)
    df = pd.DataFrame(df)
    for v in variables:
        pcnt_veh_simp(v, df)
    return df


def add_quantiles(quantiles, txdf, tx, v):
    quants = [stats.scoreatpercentile(txdf[v].values, x) for x in [0, 25, 50, 100]]
    quantiles['tx_val'] = np.concatenate([quantiles['tx_val'], quants])
    quantiles['quantile'] = np.concatenate([quantiles['quantile'], [0, 25, 50, 100]])
    quantiles['treatment'] = np.concatenate([quantiles['treatment'], [tx, tx, tx, tx]])
    quantiles['variable'] = np.concatenate([quantiles['variable'], [v, v, v, v]])


def pcnt_veh_simp(v, df):
    tx = df[f'Treatment {v}'].values
    ct = df[f'Control {v}'].values
    #ct_mean = ct.mean()
    df[f'{v} pcnt veh'] = tx / ct * 100 #(tx / ct) * 100



# -------------------
# Inside injury plots
# -------------------

def inside_and_outside_injury_barplots(
        df,
        treatment='MIPS', 
        control='DMSO (MIPS)',
        ):
    sns.set_context('paper')
    sns.set_style('ticks')
    df = pd.concat([df[df['treatment'] == treatment], df[df['treatment'] == control]])
    df['phase'] = df['time from peak count'].apply(_growth_consol)
    df['location'] = df['inside injury'].apply(_insideout_str)
    df['phase x location'] = df['phase'] + ': ' + df['location']
    res = {
        'path': [], 
        'treatment' : [], 
        'phase x location' : [], 
        'platelet count' : [], 
        'platelet density um^-3' : [], 
    }
    for k, grp in df.groupby(['path', 'treatment', 'phase x location']):
        res['path'].append(k[0])
        res['treatment'].append(k[1])
        res['phase x location'].append(k[2])
        res['platelet count'].append(grp['platelet count'].mean())
        res['platelet density um^-3'].append(grp['platelet density um^-3'].mean())
    res = pd.DataFrame(res)
    fig, axs = plt.subplots(1, 2)
    _ord = ['growth: inside injury', 'growth: outside injury', 'consolidation: inside injury', 'consolidation: outside injury']
    hue_ord = ['DMSO (MIPS)', 'MIPS']
    sns.barplot(data=res, x='phase x location', y='platelet count', hue='treatment', palette=pal1,
                order=_ord, hue_order=hue_ord, ax=axs[0], capsize=.15, linewidth=0.5, errorbar='se')
    for label in axs[0].get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')
    sns.barplot(data=res, x='phase x location', y='platelet density um^-3', hue='treatment', palette=pal1, 
                order=_ord, hue_order=hue_ord, ax=axs[1], capsize=.15, linewidth=0.5, errorbar='se')
    for label in axs[1].get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')
    sns.despine()
    fig.subplots_adjust(right=0.95, left=0.12, bottom=0.5, top=0.95, wspace=0.5, hspace=0.4)
    fig.set_size_inches(7, 3)
    plt.show()

def _insideout_str(val):
    if val:
        return 'inside injury'
    else:
        return 'outside injury'

def _growth_consol(val):
    if val > 0:
        return 'consolidation'
    else:
        return 'growth'


# -------------
# Regions plots
# -------------


def regions_abs_and_pcnt_timeplots(
        data, 
        variables,
        different_treatements=False,
        treatements=('MIPS', 'SQ', 'cangrelor'), 
        regions=('center', 'anterior', 'lateral', 'posterior'), 
        time_col='time (s)',
        hue='treatment', 
        errorbar=False,
        log=False
    ):
    sns.set_context('paper')
    sns.set_style('ticks')
    #if not log and len(variables) > 1:
    #    log = [False, ] * len(variables) 
    #if log and len(variables) > 1:
     #   log = [True, ] * len(variables) 
    if different_treatements:
        assert len(treatements) == len(variables)
    if not different_treatements:
        dfs = []
        for t in treatements:
            sdf = data[data['treatment'] == t]
            dfs.append(sdf)
        data = pd.concat(dfs).reset_index(drop=True)
    else:
        data_list = []
        for group in treatements:
            dfs = []
            for t in group:
                sdf = data[data['treatment'] == t]
                dfs.append(sdf)
            gdf = pd.concat(dfs).reset_index(drop=True)
            data_list.append(gdf)
        data = data_list
    del dfs
    if errorbar:
        es = 'bars'
        m = 'o'
    else:
        es = 'band'
        m = None
    matplotlib.rcParams.update({'font.size': 6})
    fig, axs = plt.subplots(len(regions), len(variables), sharex='col')
    plt.xticks(rotation=45)
    for j in range(len(regions)):
        r = regions[j]
        if not different_treatements:
            sdf = data[data['region'] == r]
        if len(variables) > 1:
            for i in range(len(variables)):
                if different_treatements:
                    sdf = data[i][data[i]['region'] == r]
                t_max = sdf[time_col].max()
                sdf = sdf[sdf[time_col] < t_max - 4]
                ax = axs[j, i]
                ax.set_title(r)
                sns.despine(ax=ax)
                #if log[i]:
                 #   ax.set_yscale('log')
                if time_col == 'time (s)':
                    _add_rolled(variables[i], sdf)
                sns.lineplot(data=sdf, x=time_col, y=variables[i], hue=hue, 
                             ax=ax, errorbar=("se", 1), err_style=es, marker=m, 
                             palette=pal1)
                if 'pcnt' in variables[i]:
                    ax.set_ylabel('percent vehicle (%)')
                    ax.axline((0, 100), (1, 100), color='grey', alpha=0.5)
                if time_col == 'hsec':
                    ax.set_xlabel('time post injury (s)')
                    for label in ax.get_xticklabels():
                        label.set_rotation(45)
                        label.set_ha('right')
                if time_col == 'time (s)':
                    ax.set_xlabel('time post injury (s)')
                if time_col == 'minute':
                    ax.set_xlabel('time post injury (min)')
        else:
            ax = axs[j]
            ax.set_title(r)
            sns.despine(ax=ax)
            #if log:
             #   ax.set_yscale('log')
            sns.lineplot(data=sdf, x=time_col, y=variables[0], hue=hue, ax=ax, errorbar=("se", 1), err_style=es, marker=m)
    fig.subplots_adjust(right=0.95, left=0.17, bottom=0.11, top=0.95, wspace=0.45, hspace=0.4)
    fig.set_size_inches(4.5, 7)
    #plt.xticks(rotation=45)
    plt.show()



# ----------------------
# Individual Experiments
# ----------------------

def individual_exp_inside_outside_timeplots(
        df, 
        var='platelet count', 
        treatment='MIPS', 
        control='DMSO (MIPS)'
        ):
    _add_rolled_II(var, df)
    tx = df[df['treatment'] == treatment]
    tx_i = tx[tx['inside injury'] == True]
    tx_o = tx[tx['inside injury'] == False]
    ctl = df[df['treatment'] == control]
    ctl_i = ctl[ctl['inside injury'] == True]
    ctl_o = ctl[ctl['inside injury'] == False]
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    ax0, ax1, ax2, ax3 = axs.ravel()
    print('Treatment: inside: ', tx_i.groupby('path')[var].mean())
    sns.lineplot(data=tx_i, y=var, x='time from peak count', hue='path', ax=ax0)
    ax0.set_title(f'{treatment}: inside injury')
    ax0.legend([],[], frameon=False)
    print('Treatment: outside: ', tx_o.groupby('path')[var].mean())
    sns.lineplot(data=tx_o, y=var, x='time from peak count', hue='path', ax=ax1)
    ax1.set_title(f'{treatment}: outside injury')
    ax1.legend([],[], frameon=False)
    print('Control: inside: ', ctl_i.groupby('path')[var].mean())
    sns.lineplot(data=ctl_i, y=var, x='time from peak count', hue='path', ax=ax2)
    ax2.set_title(f'{control}: inside injury')
    ax2.legend([],[], frameon=False)
    print('Control: outside: ', ctl_o.groupby('path')[var].mean())
    sns.lineplot(data=ctl_o, y=var, x='time from peak count', hue='path', ax=ax3)
    ax3.set_title(f'{control}: outside injury')
    ax3.legend([],[], frameon=False)
    sns.despine()
    fig.subplots_adjust(right=0.95, left=0.1, bottom=0.1, top=0.95, wspace=0.12, hspace=0.2)
    fig.set_size_inches(10, 7)
    plt.show()

def _add_rolled_II(col, df):
    for k, grp in df.groupby(['path', 'inside injury']):
        idxs = grp.index.values
        roll = grp[col].rolling(window=8,center=False).mean()
        df.loc[idxs, col] = roll