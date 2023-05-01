import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

MIPS_order = ['DMSO (MIPS)', 'MIPS']
cang_order = ['saline','cangrelor']#['Saline','Cangrelor','Bivalirudin']
SQ_order = ['DMSO (SQ)', 'SQ']
pal_MIPS  =dict(zip(MIPS_order, sns.color_palette('Blues')[2::3]))

pal_cang = dict(zip(cang_order, sns.color_palette('Oranges')[2::3]))

pal_SQ = dict(zip(SQ_order, sns.color_palette('Greens')[2::3]))

pal1={**pal_MIPS,**pal_cang,**pal_SQ}


def timeplots(
        save_paths,
        time_col='time (s)',
        hue='treatment', # 'path'
        names=('platelet count', 'platelet density', 'thrombus edge distance'), \
        category=None, 
        option=None,
        ):
    '''
    other_cols: tuple
        other columns to collect values for. Takes only the first value in group. 
        The first value in other_cols will be used as the hue for sns.lineplot. 
    '''
    data_list = [pd.read_csv(p) for p in save_paths]
    # plots
    if category is not None:
        data_list = [df[df[category] == option] for df in data_list]
    fig, axs = plt.subplots(1, len(names), sharex=True, sharey=False)
    for i, ax in enumerate(axs.ravel()):
        sns.lineplot(data=data_list[i], x=time_col, y=names[i], hue=hue, ax=ax, errorbar=("se", 1))
    plt.show()



def average_timebinned(
        data_paths,
        save_paths,
        time_bins=((0, 100), (100, 200), (200, 300), (300, 400), (400, 500)),
        time_col='time (s)',
        hue='treatment', 
        names=('platelet count', 'platelet density', 'thrombus edge distance'), 
        exp_col='path'
    ):
    data_list = [pd.read_csv(p) for p in data_paths]
    results = []
    for i, df in enumerate(data_list):
        res = {
            hue: [], 
            time_col: [], 
            names[i] : [], 
            exp_col : [], 
        }
        for j in range(len(time_bins)):
            sub_df = df[(df[time_col] >= time_bins[j][0]) & (df[time_col] < time_bins[j][1])]
            for k, g in sub_df.groupby([hue, exp_col]):
                res[hue].append(k[0])
                mean = g[names[i]].mean()
                res[names[i]].append(mean)
                tstring = f'{time_bins[j][0]}-{time_bins[j][1]} s'
                res[time_col].append(tstring)
                res[exp_col].append(k[1])
        res = pd.DataFrame(res)
        res.to_csv(save_paths[i])
        results.append(res)
    return results


def timebinned_boxplots(
        results, 
        data_paths=None, 
        time_col='time (s)',
        hue='treatment', 
        names=('platelet count', 'platelet density', 'thrombus edge distance'),
        ):
    if data_paths is not None:
        results = [pd.read_csv(p) for p in data_paths]
    fig, axs = plt.subplots(len(names), 1, sharex=True, sharey=False)
    for i, ax in enumerate(axs.ravel()):
        sns.boxplot(data=results[i], x=time_col, y=names[i], hue=hue, ax=ax)
    plt.show()



def area_under_the_curve():
    # coun
    pass 


def plot_three_treatements(
        df, 
        peaks,
        treatments=('MIPS', 'SQ', 'cangrelor'), 
        controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline'),
        names=('platelet count', 'platelet density'),
        pcnt_names=('platelet count pcnt', 'platelet density pcnt'),
        time_col='time (s)',
        hue='treatment', 
    ):
    sns.set_context('paper')
    sns.set_style('ticks')
    _add_time_to_peak(df, peaks, treatments, controls)
    df = df.sort_values(time_col)
    dfs = []
    for t in treatments:
        sdf = df[df['treatment'] == t]
        dfs.append(sdf)
    txdf = pd.concat(dfs).reset_index(drop=True)
    if time_col == 'time (s)':
        max_t = txdf[time_col].max()
        txdf = txdf[txdf[time_col] < max_t - 4]
    fig, axs = plt.subplots(len(names), 2, sharex=False, sharey=False)
    mips = [df[df['treatment'] == 'MIPS'], df[df['treatment'] == 'DMSO (MIPS)']]
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

if __name__ == '__main__':

    from datetime import datetime
    now = datetime.now()
    date = now.strftime("%y%m%d")

    d = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/Figure_2'
    names = [f'{date}_counts_density_outeredge_MIPS_cang_biva_tl10_oe9098_platelet count.csv', 
             f'{date}_counts_density_outeredge_MIPS_cang_biva_tl10_oe9098_platelet density.csv', 
             f'{date}_counts_density_outeredge_MIPS_cang_biva_tl10_oe9098_thrombus edge distance.csv']
    #save_paths = [os.path.join(d, n) for n in names]
    # average plots for main paper
    #timeplots(save_paths)
    # all experiments for MIPS
    #timeplots(save_paths, hue='path', category='treatment', option='MIPS')
    # all experiments for DMSO
    #timeplots(save_paths, hue='path', category='treatment', option='DMSO (MIPS)')
    #data_paths = [os.path.join(d, n) for n in names]
    #save_names = ['MIPSvsDMSO_counts_timebinned.csv', 'MIPSvsDMSO_density_timebinned.csv', 'MIPSvsDMSO_outeredge_timebinned.csv']
    #save_paths = [os.path.join(d, n) for n in save_names]
    #results = average_timebinned(data_paths, save_paths)
    #timebinned_boxplots(results)
    n = f'{date}_counts_density_outeredge_MIPS_cang_biva_tl10_oe9098.csv'
    p = os.path.join(d, n)
    df = pd.read_csv(p)

    peaks = pd.read_csv(os.path.join('/Users/abigailmcgovern/Data/platelet-analysis/MIPS/Figure_3', '230420_count-and-growth-pcnt_peaks.csv'))
    

    plot_three_treatements(df, peaks)
