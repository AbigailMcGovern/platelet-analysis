import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from plateletanalysis.variables.measure import quantile_normalise_variables
import numpy as np


def add_variable_centile_bins(
        df, 
        var, 
        pcnt_bands=((0, 25), (25, 50), (50, 75), (75, 100)), 
        names=('25th centile', '50th centile', '75th centile', '100th centile')
        ):
    col_name = var + ' centile band'
    for i, pb in enumerate(pcnt_bands):
        l, u = pb
        n = names[i]
        rdf = df[(df[var] >= l) & (df[var] < u)]
        idxs = rdf.index.values
        df.loc[idxs, col_name] = n
    return df, col_name, names



def add_xy_centre_dist(df, x_col='x_s', y_col='ys'):
    df['centre dist (um)'] = np.sqrt(df[x_col]**2 + df[y_col]**2)
    return df


def centile_binned_timeplots(df, y_cols, bin_cols):
    if isinstance(bin_cols, str):
        bin_cols = [bin_cols, ]
    df['time (s)'] = df['frame'] / 0.321764322705706
    fig, axes = plt.subplots(len(y_cols), len(bin_cols), sharex='col', sharey='row')
    sns.set_style("ticks")
    for j, bin_col in enumerate(bin_cols):
        if not bin_col.endswith('_pcnt') or not bin_col.endswith('percentile'):
            print('Quantile normalising ', bin_col)
            df = quantile_normalise_variables(df, (bin_col, ))
            bin_col = bin_col + '_pcnt'
        df, hue, band_names = add_variable_centile_bins(df, bin_col)
        for i, y_col in enumerate(y_cols):
            ax = axes[i, j]
            sns.lineplot(data=df, x='time (s)', y=y_col, hue=hue, hue_order=band_names, palette='rocket', lw=2, ax=ax)
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    #fig.subplots_adjust(right=0.5)
    plt.show()


def clot_contraction_plots(df, r_col, r_pcnt_col, dens_col, dens_pcnt_col):
    df['time (s)'] = df['frame'] / 0.321764322705706
    fig, axes = plt.subplots(2, 1, sharex=True)
    ax0, ax1 = axes.ravel()
    df, hue0, band_names0 = add_variable_centile_bins(df, r_pcnt_col)
    df, hue1, band_names1 = add_variable_centile_bins(df, dens_pcnt_col)
    sns.lineplot(data=df, x='time (s)', y=r_col, hue=hue0, hue_order=band_names0, palette='rocket', lw=1, ax=ax0)
    sns.move_legend(ax0, "upper left", bbox_to_anchor=(1, 1))
    sns.lineplot(data=df, x='time (s)', y=dens_col, hue=hue1, hue_order=band_names1, palette='rocket', lw=1, ax=ax1)
    sns.move_legend(ax1, "upper left", bbox_to_anchor=(1, 1))
    plt.show()







if __name__ == '__main__':
    d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
    sp = '/Users/amcg0011/Data/platelet-analysis/dataframes/211206_saline_df_220827_amp0.parquet'
    df = pd.read_parquet(sp)
    rename = {
        'nd15_percentile' : 'density (%)', 
        'dv' : 'dv (um/s)', 
        'phi' : 'phi (um)', 
        'theta' : 'theta (um)'
    }
    #df = df.rename(columns=rename)
    #bin_cols = ('ca_corr', 'fibrin_dist', 'rho_pcnt')
    #y_cols = [rename[key] for key in rename.keys()]
    #centile_binned_timeplots(df, y_cols, bin_cols)

    from plateletanalysis.variables.measure import quantile_normalise_variables_frame
    df = quantile_normalise_variables_frame(df, ('rho', ))
    df = quantile_normalise_variables_frame(df, ('nb_density_15', ))
    rename = {
    'rho' : 'centre distance (um)', 
    'nb_density_15' : 'local density (platelets/um^2)'
    }
    df = df.rename(columns=rename)
    clot_contraction_plots(df, 'centre distance (um)', 'rho_pcntf', 'local density (platelets/um^2)', 'nb_density_15_pcntf')