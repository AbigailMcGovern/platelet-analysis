from collections import defaultdict
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from plateletanalysis.variables.measure import quantile_normalise_variables
import numpy as np
from scipy.stats import ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols


def compare_treatments_in_time_ranges(
        df,
        bin_names=('0-10 s', '10-30 s', '30-90 s', '300-600 s'),
        lifespan_col = 'radius %',
        outlierness_col = 'Standard deviations from mean', 
        hue_col = 'treatment'
        ):
    fig, axes = plt.subplots(2, len(bin_names), sharex='col', sharey='row')
    for i, name in enumerate(bin_names):
        name = bin_names[i]
        ax0 = axes[0, i]
        ax1 = axes[1, i]
        data = df[df['time_bin'] == name]
        sns.barplot(x=hue_col, y=lifespan_col, data=data, ax=ax0, capsize=.2)
        #ax0.set_xticklabels(ax0.get_xticks(), rotation = 45)
        sns.barplot(x=hue_col, y=outlierness_col, data=data, ax=ax1, capsize=.2)
        #ax1.set_xticklabels(ax1.get_xticks(), rotation = 45)
    plt.xticks(rotation=45)
    plt.show()


def get_average_df(df):
    paths = pd.unique(df['path'])
    cols = df.columns.values
    mean_cols = [col for col in cols if isinstance(df[col].values[0], np.float64) or isinstance(df[col].values[0], np.int64)]
    obj_cols = [col for col in cols if isinstance(df[col].values[0], str)]
    all_cols = obj_cols + mean_cols
    out = {col : [] for col in all_cols}
    for k, grp in df.groupby(['path', 'time_bin']):
        for o in obj_cols:
            out[o].append(grp[o].values[0])
        for m in mean_cols:
            out[m].append(grp[m].mean())
    out = pd.DataFrame(out)
    return out



def add_time_binning_variable(
        df, 
        time_col='time (s)',
        frame_ranges=((0, 30), (30, 90), (90, 300), (300, 600)), 
        bin_names=('0-10 s', '10-30 s', '30-90 s', '300-600 s')
        ):
    for i in range(len(bin_names)):
        name = bin_names[i]
        min_f, max_f = frame_ranges[i]
        sml_df = df[(df[time_col] >= min_f) & (df[time_col] < max_f)]
        idxs = sml_df.index.values
        df.loc[idxs, 'time_bin'] = [name, ] * len(idxs)
    return df



def two_way_anova(df, vars, cat_0='treatment', cat_1='time_bin'):
    for v in vars:
        lm = f'Q("{v}") ~ {cat_0} + {cat_1} + {cat_0}:{cat_1}'
        #lm = f'Q({v}) ~ Q({loc_col}) + Q({tx_col}) + Q({loc_col}):Q({tx_col})'
        #lm = f'Q({v}) ~ C({loc_col}) + C({tx_col}) + C({loc_col}):C({tx_col})'
        model = ols(formula=lm, data=df).fit()
        res = sm.stats.anova_lm(model, typ=2)
        print(f'Two way anova for {v}: location x treatment')
        print(res)


def get_bin_means(df, lifespan_col = 'radius %', outlierness_col = 'Standard deviations from mean', tx_col='treatment'):
    out = {
        'path' : [],
        'time_bin' : [], 
        tx_col : [], 
        outlierness_col : [], 
        lifespan_col : [],
    } 
    for k, grp in df.groupby(['path', 'time_bin']):
        p, t = k
        o_mean = grp[outlierness_col].mean()
        l_mean = grp[lifespan_col].mean()
        tx = grp[tx_col].values[0]
        out['path'].append(p)
        out['time_bin'].append(t)
        out[tx_col].append(tx)
        out[lifespan_col].append(l_mean)
        out[outlierness_col].append(o_mean)
    out = pd.DataFrame(out)
    return out


def compare_means(
        df, 
        tx_0, 
        tx_1, 
        time_bin,  
        tx_col='treatment', 
        tb_col='time_bin',
        lifespan_col = 'radius %', 
        outlierness_col = 'Standard deviations from mean'
        ):
    df = df[df[tb_col] == time_bin]
    tx0_df = df[df[tx_col] == tx_0]
    tx1_df = df[df[tx_col] == tx_1]
    o_tx0 = tx0_df[outlierness_col].values
    o_tx1 = tx1_df[outlierness_col].values
    print(f't-test comparison between outlierness of {tx_0} and {tx_1} at {time_bin}')
    res = ttest_ind(o_tx0, o_tx1)
    print(res)
    l_tx0 = tx0_df[lifespan_col].values
    l_tx1 = tx1_df[lifespan_col].values
    print(f't-test comparison between lifespan of {tx_0} and {tx_1} at {time_bin}')
    res = ttest_ind(l_tx0, l_tx1)
    print(res)




if __name__ == '__main__':
    p = '/Users/amcg0011/Data/platelet-analysis/TDA/treatment_comparison/saline_biva_cang_sq_mips_PH-data-all.csv'
    #on = 'saline_biva_cang_sq_mips_outlierness.csv'
    #ln = 'saline_biva_cang_sq_mips_lifespan.csv'
    #lp = os.path.join(d, ln)
    #op = os.path.join(d, on)
    #ldf = pd.read_csv(lp)
    #odf = pd.read_csv(op)
    df = pd.read_csv(p)
    df = add_time_binning_variable(df)
    df = get_average_df(df)
    #compare_treatments_in_time_ranges(df)
    two_way_anova(df, ('radius %', 'Standard deviations from mean'))




