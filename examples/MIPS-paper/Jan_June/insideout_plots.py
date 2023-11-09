import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from toolz import curry
from plateletanalysis.analysis.plots import pal1


save_path_0 = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/Figure_3/230518_experiment_phase_insideout_onlylargeFalse.csv'
save_path_1 = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/Figure_3/230518_experiment_phase_insideout.csv'


df0 = pd.read_csv(save_path_0)
df1 = pd.read_csv(save_path_1)


def barplots_growth_vs_consol(df):
    sns.set_context('paper')
    sns.set_style('ticks')
    fig, axs = plt.subplots(2, 1, sharex=True)
    ax0, ax1 = axs.ravel()
    df['location'] = df['inside_injury'].apply(bin_inside_injury)
    df_0 = df[df['phase'] == 'growth']
    _ord = ['inside injury', 'outside injury']
    hue_ord = ['DMSO (MIPS)', 'MIPS']
    sns.barplot(data=df_0, x='location', y='platelet count', hue='treatment', palette=pal1,
                order=_ord, hue_order=hue_ord, ax=ax0, capsize=.15, linewidth=0.5, errorbar='se')
    ax0.set_title('growth')
    df_1 = df[df['phase'] == 'consolidation']
    sns.barplot(data=df_1, x='location', y='platelet count', hue='treatment', palette=pal1,
                order=_ord, hue_order=hue_ord, ax=ax1, capsize=.15, linewidth=0.5, errorbar='se')
    ax1.set_title('consolidation')
    sns.despine()
    fig.subplots_adjust(right=0.95, left=0.25, bottom=0.15, top=0.9, wspace=0.3, hspace=0.4)
    fig.set_size_inches(3, 5.5)
    plt.show()



@curry
def bin_inside_injury(val):
    if val:
        r = 'inside injury'
    else:
        r = 'outside injury'
    return r


def barplots_consolidation_large(df, vars):
    df = df[df['phase'] == 'consolidation']
    df['location'] = df['inside_injury'].apply(bin_inside_injury)
    df = df[df['location'] == 'outside injury']
    _ord = ['DMSO (MIPS)', 'MIPS']
    hue_ord = ['DMSO (MIPS)', 'MIPS']
    fig, axs = plt.subplots(len(vars), 1, sharex=True)
    for ax, v in zip(axs, vars):
        sns.barplot(data=df, x='treatment', y=v, palette=pal1, 
                    order=_ord, ax=ax, capsize=.15, 
                    linewidth=0.5, errorbar='se')
    sns.despine()
    fig.subplots_adjust(right=0.95, left=0.5, bottom=0.1, top=0.95, wspace=0.3, hspace=0.55)
    fig.set_size_inches(1.7, 7)
    plt.show()


#barplots_growth_vs_consol(df0)

variables = ('platelet count', 'platelet density gain (um^-3)', 
             'platelet average density (um^-3)', 'frame average density (um^-3)',
             'average platelet instability', 'recruitment', 
             'P(recruited < 15 s)', 'total sliding (um)', 
             'average platelet sliding (um)', 'average platelet contraction (um s^-1)', 
             'average contraction in frame (um s^-1)', 'average platelet corrected calcium',
             'average frame corrected calcium', 'shedding', 
             'average platelet tracking time (s)', 'initial corrected calcium', 
             'initial platlet density (um^-3)', 'initial platelet instability', 
             'angle at time shed', 'y coordinate at time shed', 
             'distance from centre at time shed', 'intial platelet density gain (um^-3)', 
             'P(< 15s)', 'initial platelet velocity change (um/s)')

vars = ('platelet average density (um^-3)', 'average platelet instability', 'average platelet tracking time (s)')
barplots_consolidation_large(df1, vars)
