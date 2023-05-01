import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_exps_insideout(df, var='platelet count', treatment='MIPS', control='DMSO (MIPS)'):
    _add_rolled(var, df)
    tx = df[df['treatment'] == treatment]
    tx_i = tx[tx['inside injury'] == True]
    tx_o = tx[tx['inside injury'] == False]
    ctl = df[df['treatment'] == control]
    ctl_i = ctl[ctl['inside injury'] == True]
    ctl_o = ctl[ctl['inside injury'] == False]
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    ax0, ax1, ax2, ax3 = axs.ravel()
    print('Treatment: inside: ', tx_i.groupby('path')[var].mean())
    sns.lineplot(data=tx_i, y=var, x='time from peak count', hue='path', ax=ax0)
    ax0.set_title(f'{treatment}: inside injury')
    ax0.legend([],[], frameon=False)
    print('Treatment: outside: ', tx_o.groupby('path')[var].mean())
    sns.lineplot(data=tx_o, y=var, x='time from peak count', hue='path', ax=ax1)
    ax1.set_title(f'{treatment}: outside injury')
    ax1.legend([],[], frameon=False)
    print('Control: inside: ', ctl_i.groupby('path')[var].mean())
    sns.lineplot(data=ctl_i, y=var, x='time from peak count', hue='path', ax=ax2)
    ax2.set_title(f'{control}: inside injury')
    ax2.legend([],[], frameon=False)
    print('Control: outside: ', ctl_o.groupby('path')[var].mean())
    sns.lineplot(data=ctl_o, y=var, x='time from peak count', hue='path', ax=ax3)
    ax3.set_title(f'{control}: outside injury')
    ax3.legend([],[], frameon=False)
    sns.despine()
    fig.subplots_adjust(right=0.95, left=0.1, bottom=0.1, top=0.95, wspace=0.12, hspace=0.2)
    fig.set_size_inches(10, 7)
    plt.show()

def _add_rolled(col, df):
    for k, grp in df.groupby(['path', 'inside injury']):
        idxs = grp.index.values
        roll = grp[col].rolling(window=8,center=False).mean()
        df.loc[idxs, col] = roll

sp = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/Figure_3/230428_inside_outside_counts_density.csv'
df = pd.read_csv(sp)
#plot_exps_insideout(df)
#plot_exps_insideout(df, 'platelet density um^-3')
plot_exps_insideout(df, treatment='SQ', control='DMSO (SQ)')