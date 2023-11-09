import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from plateletanalysis.variables.basic import get_treatment_name, time_seconds
from plateletanalysis.analysis.peaks_analysis import smooth_vars
from scipy import stats


# ---------
# Functions
# ---------

def donutness_comp_plots(df):
    plt.rcParams['svg.fonttype'] = 'none'
    df = df.groupby(['path', 'time (s)', 'treatment'])['donutness'].mean().reset_index()
    df = smooth_vars(df, vars=['donutness', ], gb='path', w=20)
    df = df.dropna(subset=['donutness', 'time (s)'])
    fig, ax = plt.subplots(1, 1)
    sns.lineplot(data=df, x='time (s)', y='donutness', hue='treatment', palette='rocket', ax=ax)
    sns.despine(ax=ax)
    fig.set_size_inches(3.5, 3)
    fig.subplots_adjust(right=0.97, left=0.17, bottom=0.15, top=0.9, wspace=0.4, hspace=0.4)
    plt.show()


def donutness_comp_box(df, order=['saline', 'bivalirudin', 'PAR4--']):
    vals = {tx : df[df['treatment'] == tx]['donutness magnetude'].values for tx in order}
    done = []
    for tx0 in order:
        for tx1 in order:
            n = f'{tx0} x {tx1}'
            if tx0 != tx1:
                v0 = vals[tx0]
                v1 = vals[tx1]
                res = stats.mannwhitneyu(v0, v1)
                print(n, res)
                done.append(n)
    plt.rcParams['svg.fonttype'] = 'none'
    fig, ax = plt.subplots(1, 1)
    sns.boxplot(data=df, y='donutness magnetude', x='treatment', palette='rocket', ax=ax, order=order)
    sns.stripplot(data=df, y='donutness magnetude', x='treatment', palette='rocket', ax=ax, order=order,
                  edgecolor = 'white', linewidth=0.3, jitter=True, size=5)
    sns.despine(ax=ax)
    fig.set_size_inches(3.5, 3)
    fig.subplots_adjust(right=0.97, left=0.17, bottom=0.15, top=0.9, wspace=0.4, hspace=0.4)
    plt.show()


def max_donuts(vals):
    return np.nanmax(vals)

# ---------
# Read Data
# ---------

d = '/Users/abigailmcgovern/Data/platelet-analysis/toroidal_paper/data'
names = ['bivalirudin', 'PAR4--', 'saline']
ps = [os.path.join(d, f'{name}_donut_data_scaled_sn200_n100_c50_gt1tks.csv') for name in names]
df = [pd.read_csv(p) for p in ps]
df = pd.concat(df).reset_index(drop=True)
df['treatment'] = df['path'].apply(get_treatment_name)
df = time_seconds(df)
df = df.dropna(subset=['donutness', 'time (s)'])
df[df['outlierness_mean'] == 0]['donutness'] = 0

ps0 = [os.path.join(d, f'{name}_summary_data_gt1tk.csv') for name in names]
df0 = [pd.read_csv(p) for p in ps0]
df0 = pd.concat(df0).reset_index(drop=True)
df0['treatment'] = df0['path'].apply(get_treatment_name)

# -------
# Execute
# -------

#donutness_comp_plots(df)
donutness_comp_box(df0)

# saline x bivalirudin MannwhitneyuResult(statistic=232.0, pvalue=8.589568340715082e-05)
# saline x PAR4-- MannwhitneyuResult(statistic=267.0, pvalue=0.00010303416493426825)
# bivalirudin x PAR4-- MannwhitneyuResult(statistic=138.0, pvalue=0.3370566757216328)


