import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from toolz import curry


# painfully enough, this follows on from figure_3.py... I guess sometimes life just be like that

def growth_vs_consol(
        summary_data, 
        out,
        out_g, 
        out_c, 
        treatments_line = ('MIPS', 'SQ', 'cangrelor'), 
        controls_line = ('DMSO (MIPS)', 'DMSO (SQ)', 'saline'), 
        treatments_box = ('MIPS', ), 
        controls_box = ('DMSO (MIPS)', )
        ):
    sns.set_context('paper')
    sns.set_style('ticks')
    for i, tx in enumerate(treatments_line):
        ctl = controls_line[i]
        t = out[out['treatment'] == ctl]['time peak count'].mean()
        for k, g in summary_data.groupby('treatment', group_keys=False):
        #t = out[out['treatment'] == k]['time peak count'].mean()
            if k in [ctl, tx]:
                vs = g['time (s)'].values - t
                idxs = g.index.values
                summary_data.loc[idxs, 'time from peak count (s)'] = vs
    fig, axs = plt.subplots(1, 2)
    ax0, ax1 = axs.ravel()
    add_percentage_of_control(summary_data, 'platelet count', treatments_line, controls_line, sub_with_raw=True)
    add_percentage_of_control(summary_data, 'density (platelets/um^2)', treatments_line, controls_line, sub_with_raw=True)
    print(summary_data.columns.values)
    df = pd.concat([summary_data[summary_data['treatment'] == tx] for tx in treatments_line])
    #bins_l = np.linspace(df['time from peak count (s)'].min(), df['time from peak count (s)'].max(), 20)[:-1]
    #bins_h = np.linspace(df['time from peak count (s)'].min(), df['time from peak count (s)'].max(), 20)[1:]
    #bins = (bins_l, bins_h)
    #bin_func = _s_bin(bins)
    #df['time from peak count (s)'] = df['time from peak count (s)'].apply(bin_func)

    ax0.axline((0, 100), (0, 101), color='grey', alpha=0.5, ls='--')
    ax0.axline((0, 100), (1, 100), color='grey', alpha=0.5)
    sns.lineplot(data=df, x='time from peak count (s)', y='Percent control platelet count (%)', hue='treatment', ax=ax0, ci=70)

    ax1.axline((0, 100), (0, 101), color='grey', alpha=0.5, ls='--')
    ax1.axline((0, 100), (1, 100), color='grey', alpha=0.5)
    sns.lineplot(data=df, x='time from peak count (s)', y='Percent control density (platelets/um^2) (%)', hue='treatment', ax=ax1, ci=70)

    box_df = {
        'treatment' : [], 
        'mean count' : [], 
        'phase' : [], 
    }
    for i, tx in enumerate(treatments_box):
        ctl = controls_box[i]
        txdf_g = out_g[out_g['treatment'] == tx]
        ctldf_g = out_g[out_g['treatment'] == ctl]
        txdf_c = out_c[out_c['treatment'] == tx]
        ctldf_c = out_c[out_c['treatment'] == ctl]
        growth = ['growth', ] * (len(txdf_g) + len(ctldf_g))
        consol = ['consolidation', ] * (len(txdf_c) + len(ctldf_c))
        box_df['mean count'] = np.concatenate([box_df['mean count'], 
                                               txdf_g['mean count'].values, 
                                               ctldf_g['mean count'].values,
                                                txdf_c['mean count'].values, 
                                               ctldf_c['mean count'].values])
        box_df['treatment'] = np.concatenate([box_df['treatment'], 
                                              txdf_g['treatment'].values, 
                                              ctldf_g['treatment'].values, 
                                               txdf_c['treatment'].values, 
                                               ctldf_c['treatment'].values])
        box_df['phase'] = np.concatenate([box_df['phase'], growth, consol])
    box_df = pd.DataFrame(box_df)
    #sns.boxplot(data=box_df, x='phase', y='mean count', hue='treatment', ax=ax_box)
    matplotlib.rcParams.update({'font.size': 10})
    fig.subplots_adjust(right=0.951, left=0.127, bottom=0.175, top=0.88, wspace=0.4)
    fig.set_size_inches(5, 2)
    plt.show()



def add_percentage_of_control(df, var, treatments, controls, sub_with_raw=True):
    col = f'Percent control {var} (%)'
    for i, tx in enumerate(treatments):
        ctl = controls[i]
        txdf = df[df['treatment'] == tx]
        ctldf = df[df['treatment'] == ctl]
        for k, g in txdf.groupby('time (s)', group_keys=False):
            ctl_c = ctldf[ctldf['time (s)'] == k][var].values
            ctl_c = np.nanmean(ctl_c)
            if sub_with_raw:
                n = var + ' raw'
                if ctl_c == np.NaN:
                    ctl_c = ctldf[ctldf['time (s)'] == k][n].mean()
            tx_c = g[var].mean()
            if tx_c != np.NaN:
                vs = g[var] / ctl_c * 100
            else:
                if sub_with_raw:
                    vs = g[n] / ctl_c * 100
            idxs = g.index.values
            df.loc[idxs, col] = vs


@curry
def _s_bin(bins, t):
    bl, bh = bins
    for i, l in enumerate(bl):
        h = bh[i]
        if t >= l and t < h:
            return (l + h) / 2
        


        
if __name__ == '__main__':
    sum_p = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/percentiles/count-and-growth-pcnt-3_rolling-counts.csv'
    summary_data = pd.read_csv(sum_p)
    out_p = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/percentiles/count-and-growth-pcnt-3_centile-data.csv'
    out = pd.read_csv(out_p)
    out_g_p = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/percentiles/count-and-growth-pcnt-3_centile-data-growth.csv'
    out_g = pd.read_csv(out_g_p)
    out_c_p = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/percentiles/count-and-growth-pcnt-3_centile-data-consolidation.csv'
    out_c = pd.read_csv(out_c_p)
    growth_vs_consol(summary_data, out, out_g, out_c)