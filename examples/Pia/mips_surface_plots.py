import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plateletanalysis.variables.measure import quantile_normalise_variables
from scipy.stats import bootstrap



def time_plot(df, var, positions=('surface', 'core', 'anterior_surface', 'tail', 'donut'), hue='treatment', absval=False, pcnt=False, minute_binned=False):
    fig, axes = plt.subplots(len(positions), 1, sharex=True, sharey=True)
    if len(positions) == 1:
        axes = (axes, )
    for i, ax in enumerate(axes):
        pos = positions[i]
        col = pos + f' {var}'
        if pcnt:
            old_col = pos + f' {var}'
            col = pos + f' {var} (%)'
            count_col = pos + ' count'
            df[col] = df[old_col] / df[count_col] * 100
        ax.set_title(pos)
        if absval:
            df[col] = np.absolute(df[col].values)
        if minute_binned:
            data = minute_binned_df(df, col)
            time_col = 'time (min)'
        else:
            data = df
            time_col = 'time (s)'
        sns.lineplot(x=time_col, y=col, data=data, ax=ax, hue=hue)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.show()



def experiment_wise(df, var, pos, hue='path', pcnt=False, minute_binned=False):
    if pcnt:
        old_col = pos + f' {var}'
        col = pos + f' {var} (%)'
        count_col = pos + ' count'
        df[col] = df[old_col] / df[count_col] * 100
    else:
        col = pos + f' {var}'
    if minute_binned:
        df = minute_binned_df(df, col)
        time_col = 'time (min)'
    else:
        time_col = 'time (s)'
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
    ax0, ax1 = axes.ravel()
    df0 = df[df['treatment'] == 'MIPS']
    df0 = df0.reset_index(drop=True)
    #print(df0.head())
    df1 = df[df['treatment'] == 'DMSO (MIPS)']
    df1 = df1.reset_index(drop=True)
    sns.lineplot(x=time_col, y=col, data=df0, ax=ax0, hue=hue)
    sns.move_legend(ax0, "upper left", bbox_to_anchor=(1, 1))
    sns.lineplot(x=time_col, y=col, data=df1, ax=ax1, hue=hue)
    sns.move_legend(ax1, "upper left", bbox_to_anchor=(1, 1))
    plt.show()



def minute_binned_df(df, col):
    values = []
    time = []
    treatment = []
    path = []
    for k, g in df.groupby(['treatment', 'path']):
        mins = [[0, i * 60] for i in range(1, 11)]
        minutes = np.array([i for i in range(1, 11)]).astype(float)
        vals = [g[(g['time (s)'] > m[0]) & (g['time (s)'] < m[1])][col].mean() for m in mins]
        values = np.concatenate([values, vals])
        time = np.concatenate([time, minutes])
        tx = [k[0], ] * len(minutes)
        treatment = np.concatenate([treatment, tx])
        p = [k[1], ] * len(minutes)
        path = np.concatenate([path, p])
    out = {
        'treatment' : treatment,
        'path' : path,
        'time (min)' : time, 
        col : values
    }
    out = pd.DataFrame(out)
    return out



def boostrap_CIs(df, var, pos='surface', time_bins=((120, 260), (300, 600))):
    col = pos + f' {var}'
    treatments = pd.unique(df['treatment'])
    for tb in time_bins:
        smldf = df[(df['time (s)'] > tb[0]) & (df['time (s)'] < tb[1])]
        print(f'Getting results for time bin {tb[0]} - {tb[1]} seconds:')
        for tx in treatments:
            smlerdf = smldf[smldf['treatment'] == tx]
            data = []
            for k, g in smlerdf.groupby('path'):
                data.append(g[col].mean())
            data = (data, )
            bootstrap_ci = bootstrap(data, np.median, confidence_level=0.95,
                             random_state=1, method='percentile')
            print(f'CI for {tx}')
            print(bootstrap_ci.confidence_interval)




if __name__ == '__main__':
    save_path = '/Users/amcg0011/Data/platelet-analysis/MIPS_surface/mips_dmso_positional_data-new.csv'
    df = pd.read_csv(save_path)
    df = df[df['treatment'] != 'DMSO (salgav)']
    df = df[df['frame'] < 190]
    # exclude noisy/strange thrombi
    df = df[df['path'] != '210721_IVMTR129_Inj8_galsavDMSO_exp3'] # shouldn't be used for MIPS
    df = df[df['path'] != '210721_IVMTR129_Inj7_galsavDMSO_exp3'] # shouldn't be used for MIPS
    df = df[df['path'] != '210505_IVMTR103_Inj4_MIPS_exp3'] # tiny poorly formed clot with no surface 210511_IVMTR104_Inj4_MIPS_exp3
    df = df[df['path'] != '210511_IVMTR104_Inj4_MIPS_exp3'] 
    #time_plot(df, 'density (platelets/um^2)', positions=('surface', 'core', 'anterior_surface', 'tail', 'donut'), hue='treatment')
    #time_plot(df, 'dv (um/s)', positions=('surface', 'core', 'anterior_surface', 'tail', 'donut'), hue='treatment', absval=False)
    #experiment_wise(df, 'dvz (um/s)', 'surface', hue='path')
    #boostrap_CIs(df, 'density (platelets/um^2)', pos='core', time_bins=((0, 120), (120, 260), (300, 400), (400, 600)))
    #boostrap_CIs(df, 'dv (um/s)', pos='surface', time_bins=((0, 30), (0, 120), (120, 260), (300, 400), (400, 600)))
    #time_plot(df, 'centre distance', positions=('surface', 'core', 'anterior_surface', 'tail', 'donut'), hue='treatment', absval=False)
    time_plot(df, 'corrected calcium', positions=('surface', 'core', 'anterior_surface', 'tail', 'donut'), hue='treatment', absval=False)
    #time_plot(df, 'gained', positions=('surface', 'core', 'anterior_surface', 'tail', 'donut'), hue='treatment', absval=False)
    #time_plot(df, 'lost', positions=('surface', 'core', 'anterior_surface', 'tail', 'donut'), hue='treatment', absval=False)
    #time_plot(df, 'dvy (um/s)', positions=('surface', 'core', 'anterior_surface', 'tail', 'donut'), hue='treatment', absval=False)
    #boostrap_CIs(df, 'dvy (um/s)', pos='surface', time_bins=((0, 120), (120, 260), (200, 300), (300, 400), (400, 600)))
    #time_plot(df, 'contraction (um/s)', positions=('surface', 'core', 'anterior_surface', 'tail', 'donut'), hue='treatment', absval=False)
    #boostrap_CIs(df, 'contraction (um/s)', pos='surface', time_bins=((0, 120), (120, 180), (180, 240), (240, 300), (200, 300), (300, 400), (400, 600)))
    #time_plot(df, 'to core', positions=('surface', 'anterior_surface'), hue='treatment', absval=False, pcnt=True)
    #time_plot(df, 'stable', positions=('surface', 'core', 'anterior_surface', 'tail', 'donut'), hue='treatment', absval=False, pcnt=True)
    #time_plot(df, 'to surface', positions=('core', ), hue='treatment', absval=False, pcnt=True)
    #experiment_wise(df, 'to surface', 'core', hue='path')
    #experiment_wise(df, 'count', 'core', hue='path')
    #experiment_wise(df, 'to surface', 'core', hue='path', pcnt=True, minute_binned=True)
    #time_plot(df, 'to surface', positions=('core', ), hue='treatment', absval=False, pcnt=True, minute_binned=True)
    #time_plot(df, 'to core', positions=('surface', 'anterior_surface'), hue='treatment', absval=False, pcnt=True, minute_binned=True)
    #time_plot(df, 'dvz (um/s)', positions=('surface', 'core', 'anterior_surface', 'tail', 'donut'), hue='treatment', absval=False, pcnt=False, minute_binned=True)