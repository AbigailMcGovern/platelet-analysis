from plateletanalysis.topology.donutness import largest_loop_comparison_data, plot_donut_comparison
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns



sd = '/Users/amcg0011/Data/platelet-analysis/TDA/treatment_comparison'
save_path = os.path.join(sd, '221025_longest-loop-analysis.csv')
data = pd.read_csv(save_path)
#print(pd.unique(data['treatment']))
# ['saline' 'bivalirudin' 'cangrelor' 'SQ' 'MIPS' 'control' 'PAR4--'
 # 'PAR4-- bivalirudin' 'salgav' 'DMSO (salgav)' 'DMSO (SQ)']


def plot_experiments(data, treatment):
    fig, axs = plt.subplots(2, 1, sharex=True)
    ax0, ax1 = axs.ravel()
    df = data[data['treatment'] == treatment]
    sns.lineplot(y='persistence (%) rolling', x='time (s)', ax=ax0, data=df, hue='path')
    sns.move_legend(ax0, "upper left", bbox_to_anchor=(1, 1))
    sns.lineplot(y='difference from mean (std dev) rolling', x='time (s)', ax=ax1, data=df, hue='path')
    sns.move_legend(ax1, "upper left", bbox_to_anchor=(1, 1))
    plt.show()


def rolling_variable(df, p='path', t='time (s)', y='difference from mean (std dev)'):
    n = y + ' rolling'
    for k, g in df.groupby([p]):
        g = g.sort_values(t)
        idx = g.index.values
        rolling = g[y].rolling(window=20,win_type='bartlett',min_periods=3,center=False).mean()
        df.loc[idx, n] = rolling
    return df


# ----------------------
# Make comparitive plots
# ----------------------

#plot_1 = ('saline', 'cangrelor', 'bivalirudin')
#plot_donut_comparison(data, plot_1)

#plot_2 = ('DMSO (MIPS)', 'MIPS')
#plot_donut_comparison(data, plot_2)

#plot_3 = ('SQ', 'DMSO (SQ)')
#plot_donut_comparison(data, plot_3)

data = data[data['frame'] < 190]
data = rolling_variable(data)
data = rolling_variable(data, y='persistence (%)')
#plot_4 = ('saline', 'bivalirudin', 'PAR4--', 'PAR4-- bivalirudin')
#plot_donut_comparison(data, plot_4)

#plot_5 = ('salgav', 'DMSO (salgav)')
#plot_donut_comparison(data, plot_5)

#plot_6 = ('saline', 'control', 'DMSO (MIPS)', 'DMSO (SQ)', 'DMSO (salgav)')
#plot_donut_comparison(data, plot_6)


plot_experiments(data, 'DMSO (SQ)')