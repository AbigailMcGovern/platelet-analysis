import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats


def violin_plots(df, vars, exp_col):
    sns.set_style("ticks")
    fig, axes = plt.subplots(len(vars), 1, sharex=True)
    for i, ax in enumerate(axes.ravel()):
        sns.violinplot(exp_col, vars[i], ax=ax, palette="husl", data=df)
    plt.xticks(rotation=45)
    fig.subplots_adjust(bottom=0.3)
    plt.show()


def kde_plots_for_comparison(df, dv_col, r0_col, r1_col, r2_col, frame_range=(115, 193)):
    df = df[(df['frame'] > frame_range[0]) & (df['frame'] < frame_range[1])]
    sns.set_style("ticks")
    y = df[dv_col].values
    x0 = df[r0_col].values
    x1 = df[r1_col].values
    x2 = df[r2_col].values
    fig, axes = plt.subplots(1, 3, sharey=True)
    ax0, ax1, ax2 = axes.ravel()
    # plot 0
    slope, intercept = _stats_with_print(x0, y, r0_col)
    sns.kdeplot(r0_col, dv_col, data=df, palette='mako', ax=ax0, fill=True, cmap='rocket_r')
    ax0.axline((0, intercept), (1, intercept + slope), ls='--', c='black')
    # plot 1
    slope, intercept = _stats_with_print(x1, y, r1_col)
    sns.kdeplot(r1_col, dv_col, data=df, palette='mako', ax=ax1, fill=True, cmap='rocket_r')
    ax1.axline((0, intercept), (1, intercept + slope), ls='--', c='black')
    # plot 2
    slope, intercept = _stats_with_print(x2, y, r2_col)
    sns.kdeplot(r2_col, dv_col, data=df, palette='mako', ax=ax2, fill=True, cmap='rocket_r')
    ax2.axline((0, intercept), (1, slope), ls='--', c='black')
    plt.show()


def statistical_comparison(df, dv_col, r0_col, r1_col, r2_col, frame_range=(115, 193)):
    df = df[(df['frame'] > frame_range[0]) & (df['frame'] < frame_range[1])]
    y = df[dv_col].values
    x0 = df[r0_col].values
    x1 = df[r1_col].values
    x2 = df[r2_col].values
    _stats_with_print(x0, y, r0_col)
    _stats_with_print(x1, y, r1_col)
    _stats_with_print(x2, y, r2_col)


def _stats_with_print(x, y, col):
    print(f'Pearson R for {col}')
    r, p = stats.pearsonr(x, y)
    print(f'r = {r}, p = {p}')
    print(f'Linear regression for {col}')
    slope, intercept, r, p, se = stats.linregress(x, y)
    rsq = r ** 2
    print(f'slope = {slope}, intercept = {intercept}, r = {r}, r squared = {rsq}, se = {se}, p = {p}')
    return slope, intercept



if __name__ == '__main__':
    import pandas as pd
    d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
    sp = '/Users/amcg0011/Data/platelet-analysis/dataframes/211206_saline_df_220822_amp0.parquet'
    df = pd.read_parquet(sp)
    names = {
        'x_s' : 'x (um)', 
        'ys' : 'y (um)', 
        'zs' : 'z (um)', 
       # 'phi' : 'phi (rad)', 
       # 'theta' : 'theta (rad)', 
        'rho' : 'rho (um)', 
       #'x_s_pcnt' : 'x (%)'
        }
    df = df.rename(columns=names)
    df = df.rename(columns={'path' : 'injury', 'dv' : 'dv (um/s)', 'rho_pcnt' : 'rho (%)'})
    #vars = [names[key] for key in names.keys()]
    #violin_plots(df, vars, 'injury')
    for key, grp in df.groupby(['injury', ]):
        idx = grp.index.values
        max_val = grp['rho (um)'].max()
        df.loc[idx, 'P(rho max)'] = grp['rho (um)'] / max_val
    
    df = df.dropna(subset=['dv (um/s)', 'rho (um)', 'rho (%)', 'P(rho max)'])
    kde_plots_for_comparison(df, 'dv (um/s)', 'rho (um)', 'rho (%)', 'P(rho max)', frame_range=(115, 193))
    #statistical_comparison(df, 'dv (um/s)', 'rho (um)', 'rho (%)', 'P(rho max)', frame_range=(115, 193))

