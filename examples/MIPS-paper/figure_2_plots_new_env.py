import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns




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
        treatements=('MIPS', 'SQ', 'cangrelor'), 
        names=('platelet count pcnt', 'platelet density pcnt', 'thrombus edge distance pcnt'),
        time_col='time (s)',
        hue='treatment', 
    ):
    dfs = []
    for t in treatements:
        sdf = df[df['treatment'] == t]
        dfs.append(sdf)
    df = pd.concat(dfs).reset_index(drop=True)
    fig, axs = plt.subplots(1, len(names), sharex=True, sharey=False)
    for i, ax in enumerate(axs.ravel()):
        sns.lineplot(data=df, x=time_col, y=names[i], hue=hue, ax=ax, errorbar=("se", 1))
    plt.show()



if __name__ == '__main__':
    d = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/Figure_2'
    names = ['MIPSvsDMSO_counts.csv', 'MIPSvsDMSO_density.csv', 'MIPSvsDMSO_outeredge.csv']
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
    n = 'counts_density_outeredge_MIPS_cang_biva_tl10_oe9098.csv'
    p = os.path.join(d, n)
    df = pd.read_csv(p)
    plot_three_treatements(df)
