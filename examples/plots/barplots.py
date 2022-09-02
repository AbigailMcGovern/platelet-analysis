import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from plateletanalysis.variables.measure import quantile_normalise_variables
import numpy as np
from scipy import stats


def compare_treatments_in_time_ranges(
        ldf,
        odf, 
        frame_ranges=((0, 30), (30, 90), (90, 300), (300, 600)), 
        bin_names=('0-10 s', '10-30 s', '30-90 s', '300-600 s'),
        time_col = 'time (s)',
        lifespan_col = 'radius %',
        outlierness_col = 'Standard deviations from mean', 
        hue_col = 'treatment'
        ):
    ldf = add_time_binning_variable(ldf, time_col, frame_ranges, bin_names)
    odf = add_time_binning_variable(odf, time_col, frame_ranges, bin_names)
    fig, axes = plt.subplots(2, len(bin_names), sharex='col', sharey='row')
    for i, name in enumerate(bin_names):
        name = bin_names[i]
        ax0 = axes[0, i]
        ax1 = axes[1, i]
        l_data = ldf[ldf['time_bin'] == name]
        o_data = odf[odf['time_bin'] == name]
        sns.barplot(x=hue_col, y=lifespan_col, data=l_data, ax=ax0, capsize=.2)
        #ax0.set_xticklabels(ax0.get_xticks(), rotation = 45)
        sns.barplot(x=hue_col, y=outlierness_col, data=o_data, ax=ax1, capsize=.2)
        #ax1.set_xticklabels(ax1.get_xticks(), rotation = 45)
    plt.xticks(rotation=45)
    plt.show()



def add_time_binning_variable(df, time_col, frame_ranges, bin_names):
    for i in range(len(bin_names)):
        name = bin_names[i]
        min_f, max_f = frame_ranges[i]
        sml_df = df[(df[time_col] >= min_f) & (df[time_col] < max_f)]
        idxs = sml_df.index.values
        df.loc[idxs, 'time_bin'] = [name, ] * len(idxs)
    return df


if __name__ == '__main__':
    d = '/Users/amcg0011/Data/platelet-analysis/TDA/treatment_comparison'
    on = 'saline_biva_cang_sq_mips_outlierness.csv'
    ln = 'saline_biva_cang_sq_mips_lifespan.csv'
    lp = os.path.join(d, ln)
    op = os.path.join(d, on)
    ldf = pd.read_csv(lp)
    odf = pd.read_csv(op)
    compare_treatments_in_time_ranges(ldf, odf, frame_ranges=((0, 30), (30, 90), (90, 300), (300, 600)), 
                                      bin_names=('0-10 s', '10-30 s', '30-90 s', '300-600 s'), time_col = 'time (s)',
                                      lifespan_col = 'radius %', outlierness_col = 'Standard deviations from mean', 
                                      hue_col = 'treatment')




