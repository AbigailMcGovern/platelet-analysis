import enum
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
    df['xy centre dist (um)'] = np.sqrt(df[x_col]**2 + df[y_col]**2)
    return df


def centile_binned_timeplot(df, y_cols, bin_col, do_pcnt=True):
    fig, axes = plt.subplots(len(y_cols),1, sharex=True)
    sns.set_style("ticks")
    if do_pcnt:
        df = quantile_normalise_variables(df, (bin_col, ))
        bin_col = bin_col + '_pcnt'
    for i, ax in enumerate(axes.ravel()):
        y_col = y_cols[i]
        df, hue, band_names = add_variable_centile_bins(df, bin_col)
        df['time (s)'] = df['frame'] / 0.321764322705706
        sns.lineplot(data=df, x='time (s)', y=y_col, hue=hue, hue_order=band_names, palette='rocket', lw=2, ax=ax)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        ax.axline((0, 0), (1, 0), ls='--', c='black')
    fig.subplots_adjust(right=0.8)
    plt.show()




if __name__ == '__main__':
    import pandas as pd
    d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
    sp = '/Users/amcg0011/Data/platelet-analysis/dataframes/211206_saline_df_220827_amp0.parquet'
    df = pd.read_parquet(sp)
    df = df[df['nrtracks'] > 10]
    df['dv phi (rad/s)'] = df['phi_diff'] * 0.321764322705706
    df['local dv phi (rad/s)'] = df['nb_phi_diff_15'] * 0.321764322705706
    names = {
        'dvz' : 'dv z (um/s)', 
        'nb_dvz_15' : 'local dv z (um/s)', 
        'nb_dvy_15' : 'local dv y (um/s)', 
    }
    df = df.rename(columns=names)
    df = add_xy_centre_dist(df)
    #centile_binned_timeplot(df, ('dv z (um/s)', 'dv phi (rad/s)'), 'xy centre dist (um)', do_pcnt=True)
    #centile_binned_timeplot(df, ('local dv z (um/s)', 'local dv phi (rad/s)'), 'xy centre dist (um)', do_pcnt=True)
    #centile_binned_timeplot(df, ('local dvz (um/s)', 'local d phi (rad/s)'), 'rho_pcnt', do_pcnt=False)
    #centile_binned_timeplot(df, ('dv z (um/s)', 'dv phi (rad/s)'), 'xy centre dist (um)', do_pcnt=True)
    centile_binned_timeplot(df, ('local dv z (um/s)', 'local dv y (um/s)'), 'xy centre dist (um)', do_pcnt=True)
    
