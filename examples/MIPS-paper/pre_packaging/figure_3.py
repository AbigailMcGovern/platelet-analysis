import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
from plateletanalysis.variables.measure import quantile_normalise_variables
from plateletanalysis.variables.transform import spherical_coordinates
from plateletanalysis.variables.basic import get_treatment_name
from toolz import curry
from scipy.signal import find_peaks #, peak_widths, peak_prominences
from pathlib import Path
import seaborn as sns
from time import time
import matplotlib
from sklearn.preprocessing import StandardScaler
from plateletanalysis.variables.basic import size_var, inside_injury_var


# -----
# CMAPS
# -----

MIPS_order = ['DMSO (MIPS)', 'MIPS']
cang_order = ['saline','cangrelor']#['Saline','Cangrelor','Bivalirudin']
SQ_order = ['DMSO (SQ)', 'SQ']
pal_MIPS  = dict(zip(MIPS_order, sns.color_palette('Blues')[2::3]))
pal_cang = dict(zip(cang_order, sns.color_palette('Oranges')[2::3]))
pal_SQ = dict(zip(SQ_order, sns.color_palette('Greens')[2::3]))
pal1 = {**pal_MIPS,**pal_cang,**pal_SQ}

# -----
# Funcs
# -----


def add_time_seconds(df, frame_col='frame'):
    df['time (s)'] = df[frame_col] / 0.321764322705706
    return df


def add_cylr(df):
    df['cyl_r'] = ((df['x_s'] ** 2) + (df['ys'] ** 2)) ** 0.5
    return df


def _exclude_back_quadrant(df, col='phi', lim=- np.pi / 2):
    df = df[df[col] > lim]
    return df


def add_outeredge_var(df, pcnt_lims=(90, 98)):
    df['outer_edge'] = False
    sml_df = _exclude_back_quadrant(df)
    sml_df = quantile_normalise_variables(sml_df, ('dist', ))
    sml_df = sml_df[(sml_df['dist_pcnt'] > pcnt_lims[0]) & (sml_df['dist_pcnt'] <= pcnt_lims[1])]
    idxs = sml_df.index.values
    df.loc[idxs, 'outer_edge'] = True
    return df


def count(grp):
    return len(grp)


def density(grp):
    return np.mean(grp['nd_density_15'].values)


def outeredge(grp):
    return np.mean(grp[grp['outer_edge'] == True]['dist'].values)



def add_rolling_counts(df : pd.DataFrame):
    gb = ['path', 'time (s)']
    df = df.sort_values('time (s)')
    counts = df.groupby(gb)['particle'].apply(count)
    df = df.set_index(gb)
    #df['count'] = counts
    counts = counts.reset_index()
    rolling_counts = counts.groupby('path')['particle'].rolling(window=8, min_periods=1,center=False).mean()
    df['rolling_count'] = rolling_counts
    #df = df.set_index(gb)
    df = df.reset_index()
    return df


def find_max_growth_and_size(df: pd.DataFrame, save_path, insideout):
    #df = add_rolling_counts(df)
    #df['rolling_growth'] = df.groupby('path')['rolling_count'].diff() * 0.321764322705706
    peaks_data = {
        'path' : [], 
        'treatment' : [],
        'peak count' : [], 
        'peak growth' : [], 
        'time peak count' : [],
        'time trough growth' : []
    }
    summary_data = {
        'path' : [], 
        'treatment' : [], 
        'time (s)' : [],
        'platelet count raw' : [], 
        #'growth (platelets/s)' : [], 
        'size (um) raw' : [], 
        'density (platelets/um^2) raw' : [], 
        'thrombus size' : [], 
    }
    if insideout:
        summary_data['inside injury'] = []
    print('Obtaining counts...')
    df = size_var(df)
    if insideout:
        gb = ['path', 'time (s)', 'treatment', 'inside_injury']
    else:
        gb = ['path', 'time (s)', 'treatment']
    for k, grp in df.groupby(gb):
        summary_data['path'].append(k[0])
        summary_data['time (s)'].append(k[1])
        summary_data['treatment'].append(k[2])
        summary_data['platelet count raw'].append(len(grp))
        sz_df = grp[grp['outer_edge'] == True] # cyl_r 90-98th centile plts
        edge = sz_df['cyl_r'].mean()
        dens = grp['nb_density_15'].mean()
        summary_data['size (um) raw'].append(edge)
        summary_data['density (platelets/um^2) raw'].append(dens)
        summary_data['thrombus size'].append(grp['size'].values[0])
        if insideout:
            summary_data['inside injury'].append(grp['inside_injury'].values[0])
        #summary_data['growth (platelets/s)'].append(grp['rolling_growth'].values[0])
    summary_data = pd.DataFrame(summary_data)
    summary_data = summary_data.sort_values('time (s)') # sorted according to time ... important to roll
    print('Getting rolling counts and growth...')
    for k, grp in summary_data.groupby('path'):
        #grp = grp.sort_values('time (s)')
        _add_rolled_and_diff(grp, 'platelet count raw', 'growth (platelets/s)', summary_data)
        _add_rolled_and_diff(grp, 'size (um) raw', 'growth (um/s)', summary_data)
        _add_rolled_and_diff(grp, 'density (platelets/um^2) raw', 'contraction (platelets/um^2/s)', summary_data)
    print('Finding peaks...')
    for k, grp in summary_data.groupby('path'):
        count_peak_idxs = find_peaks(grp['platelet count'].values)
        growth_peak_idxs = find_peaks(grp['growth (platelets/s)'].values)
        neg_growth_peak_idxs = find_peaks(- grp['growth (platelets/s)'].values)
        if len(count_peak_idxs[0]) > 0:
            count_t = grp['time (s)'].values[count_peak_idxs[0][0]]
            count_v = grp['platelet count'].values[count_peak_idxs[0][0]]
        else:
            count_t = np.NaN
            count_v = np.NaN
        if len(growth_peak_idxs[0]) > 0:
            growth_t = grp['time (s)'].values[neg_growth_peak_idxs[0][0]]
            growth_v = grp['growth (platelets/s)'].values[growth_peak_idxs[0][0]]
        else:
            growth_t = np.NaN
            growth_v = np.NaN
        peaks_data['path'].append(k)
        peaks_data['peak count'].append(count_v)
        peaks_data['peak growth'].append(growth_v)
        peaks_data['time peak count'].append(count_t)
        peaks_data['time trough growth'].append(growth_t)
        peaks_data['treatment'].append(grp['treatment'].values[0])
        idxs = grp.index.values
        summary_data.loc[idxs, 'time peak count'] = count_t
        summary_data.loc[idxs, 'time  trough growth'] = growth_t
    peaks_data = pd.DataFrame(peaks_data)
    save_peaks = os.path.join(Path(save_path).parents[0], Path(save_path).stem + '_peaks.csv')
    peaks_data.to_csv(save_peaks)
    #paths = pd.unique(summary_data['path'])
    #print('')
    #for k, grp in df.groupby('path'):
      #   pdf = peaks_data[peaks_data['path'] == k]
      #  g_t = pdf['time peak growth']
      #  c_t = pdf['time peak count']
      #  cd = grp[grp['time (s)'] == c_t]
      #  gd = grp[grp['time (s)'] == g_t]
      #  cidxs = cd.index.values
      #  gidxs = gd.index.values
      #  df.loc[gidxs, 'max_growth'] = True
      #  df.loc[cidxs, 'max_count'] = True
    #g_df = df[df['max_growth'] == True]
    #c_df = df[df['max_count'] == True]
    save_growth_plots(summary_data, peaks_data, save_path)
    save_summary = os.path.join(Path(save_path).parents[0], Path(save_path).stem + '_rolling-counts.csv')
    summary_data.to_csv(save_summary)
    return summary_data, peaks_data


def _add_rolled_and_diff(grp, col, diff_n, summary_data):
    idxs = grp.index.values
    roll = grp[col].rolling(window=20,center=False).mean()
    coln = col[:-4]
    summary_data.loc[idxs, coln] = roll
    #grp[diff_n] = roll.diff() * 0.321764322705706
    summary_data.loc[idxs, diff_n] = roll.diff().rolling(window=20,center=False).mean() * 0.321764322705706


def save_growth_plots(summary_data, peaks_data, save_path):
    #treatments = pd.unique(summary_data['treatment'])
    fig, ax = plt.subplots(1, 1)
    for tx, grp in summary_data.groupby('treatment'):
        sp = os.path.join(Path(save_path).parents[0], Path(save_path).stem + f'_{tx}-count-plot.pdf')
        pk = peaks_data[peaks_data['treatment'] == tx]
        sns.lineplot(data=grp, x='time (s)', y='platelet count', hue='path', ax=ax)
        x, y = pk['time peak count'].values, pk['peak count'].values
        ax.scatter(x, y, s=7)
        fig.savefig(sp)
        ax.clear()
        sp = os.path.join(Path(save_path).parents[0], Path(save_path).stem + f'_{tx}-growth-plot.pdf')
        sns.lineplot(data=grp, x='time (s)', y='growth (platelets/s)', ax=ax, hue='path')
        #ax.scatter(x=pk['time peak growth'].values, y=pk['peak growth'].values, s=7)
        fig.savefig(sp)
        ax.clear()

    
def _count_max(grp, col):
    return grp[col].max()

def _growth_max(grp, col):
    return grp[col].max()

def _AUC_count(grp, col):
    return grp[col].sum()

def _count_mean(grp, col):
    return grp[col].mean()

def _time_to_peak(grp, col):
    t = grp[col].values[0]
    return t

def _growth_mean(grp, col):
    return grp[col].mean()




def centile_score_data(
        df, 
        save_path,
        insideout=False
        ):
    df = df[df['nrtracks'] > 1]
    summary_data, peaks_data = find_max_growth_and_size(df, save_path, insideout)
    groupby = ['path', 'treatment',]
    if insideout:
        groupby.append('inside injury')
    funcs = [_count_max, _growth_max, _AUC_count, 
             _count_mean, _time_to_peak, _growth_mean,

             _count_max, _growth_max, _AUC_count, 
             _count_mean,  

             _count_max, _growth_max, _AUC_count, 
             _count_mean, 
             ]
    func_names = ['max count', 'max growth (platelets/s)', 'count AUC', 
                  'mean count', 'time max count', 'mean growth (platelets/s)',

                  'max size (um)', 'max growth (um/s)', 
                  'size AUC (um^2)', 'mean size (um)', 
                  
                  'max density (platelets/um^2)', 'max contraction (platelets/um^2/s)',
                  'density AUC (platelets/um^4)', 'mean density (platelets/um^2)', 
                  ]
    
    apply_cols = ['platelet count', 'growth (platelets/s)', 'platelet count', 
                  'platelet count', 'time peak count', 'growth (platelets/s)',

                  'size (um)', 'growth (um/s)', 
                  'size (um)', 'size (um)', 
                  
                  'density (platelets/um^2)', 'contraction (platelets/um^2/s)', 
                  'density (platelets/um^2)', 'density (platelets/um^2)', ]
    out = groupby_apply_all(summary_data, apply_cols, funcs, func_names, groupby)
    summary_data_g = summary_data[summary_data['time (s)'] < summary_data['time peak count']]
    out_g = groupby_apply_all(summary_data_g, apply_cols, funcs, func_names, groupby)
    summary_data_c = summary_data[summary_data['time (s)'] > summary_data['time peak count']]
    out_c = groupby_apply_all(summary_data_c, apply_cols, funcs, func_names, groupby)
    for func in func_names:
        out = centile_of_score_grouped(out, func, insideout)
    sp = os.path.join(Path(save_path).parents[0], Path(save_path).stem + f'_centile-data.csv')
    out.to_csv(sp)
    sp = os.path.join(Path(save_path).parents[0], Path(save_path).stem + f'_centile-data-growth.csv')
    out_g.to_csv(sp)
    sp = os.path.join(Path(save_path).parents[0], Path(save_path).stem + f'_centile-data-consolidation.csv')
    out_c.to_csv(sp)
    return out, out_g, out_c, summary_data, peaks_data


def centile_of_score_grouped(out, col, insideout):
    n = col + ' pcnt'
    if insideout:
        gb = ['treatment', 'inside injury']
    else:
        gb = 'treatment'
    for k, grp in out.groupby(gb):
        def func(val):
            group_data = grp[col].values
            pcnt = stats.percentileofscore(group_data, val)
            return pcnt
        vals = grp[col].apply(func)
        idx = grp.index.values
        out.loc[idx, n] = vals
    return out




def groupby_apply_all(df, apply_cols, funcs, func_names, groupby):
    '''
    For each group in the groupby (e.g., group = one injury @ one timepoint), 
    apply a list of functions (func). Each function should return a single value from 
    the group, which will added to the function's column (func_names). Each group 
    will contribute a single row in the output dataframe.

    - clunky with a for loop but perfectly reasonable for low numbers of groups
    - probably dont use when grouping by injury X particle ID (platelet tracking ID)
    '''
    t = time()
    print(f'grouping by {groupby}')
    print(f'applying functions to obtain {func_names}')
    out = {col : [] for col in groupby} 
    for n in func_names:
        out[n] = []
    apply_col_u = list(set(apply_cols))
    for col in apply_col_u:
        out[col] = []
    for k, grp in df.groupby(groupby):
        for i, col in enumerate(groupby):
            out[col].append(k[i])
        for col in apply_col_u:
            out[col].append(grp[col].mean())
        for func, name, col in zip(funcs, func_names, apply_cols):
            res = func(grp, col) # res is a scalar
            out[name].append(res)
    out = pd.DataFrame(out)
    t = time() - t
    print(f'Took {t} seconds')
    return out


def add_time_seconds(df, frame_col='frame'):
    df['time (s)'] = df[frame_col] / 0.321764322705706
    return df


def pp_plot(
        data, 
        variables, 
        treatments=('MIPS', 'SQ', 'cangrelor'), 
        controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline')
        ):
    fig, axs = plt.subplots(len(variables), len(treatments))
    for i, v in enumerate(variables):
        for j, tx in enumerate(treatments):
            ax = axs[i, j]
            var_pcnt = v + ' pcnt'
            txdf = data[data['treatment'] == tx]
            ctrldf = data[data['treatment'] == controls[j]]
            df = pd.concat([txdf, ctrldf])
            sns.lineplot(data=df, x=var_pcnt, y=v, ax=ax, hue='treatment')
            ax.set_title(tx)
    plt.show()
    fig.clear()
    fig, axs = plt.subplots(len(variables), len(treatments))
    for i, v in enumerate(variables):
        for j, tx in enumerate(treatments):
            ax = axs[i, j]
            var_pcnt = v + ' pcnt'
            txdf = data[data['treatment'] == tx]
            ctrldf = data[data['treatment'] == controls[j]]
            #df = pd.concat([txdf, ctrldf])
            if len(txdf) < len(ctrldf):
                n = len(txdf)
            else:
                n = len(ctrldf)
            pcnt = np.linspace(0, 100, n)
            tx_score = [stats.scoreatpercentile(txdf[v].values, p) for p in pcnt]
            ctrl_score = [stats.scoreatpercentile(ctrldf[v].values, p) for p in pcnt]
            df = {
                f'Control {v}' : tx_score, 
                f'Treatment {v}' : ctrl_score, 
                'Percentile' : pcnt
            }
            df = pd.DataFrame(df)
            sns.scatterplot(x=f'Control {v}', y=f'Treatment {v}', data=df, ax=ax, hue='Percentile')
            sns.lineplot(x=f'Control {v}', y=f'Control {v}', data=df, ax=ax)
    plt.show()



def comparative_pp_plots(
        data, 
        variables, 
        save_path,
        treatments=('MIPS', 'SQ', 'cangrelor'), 
        controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline'), 
        insideout=False
        ):
    sns.set_context('paper')
    sns.set_style('ticks')
    # generate data
    dfs = []
    quantiles = {
        'quantile' : [], 
        'treatment' : [], 
        'tx_val' : [], 
        'variable' : [], 
    }
    if insideout:
        quantiles['inside injury'] = []
    for i, tx in enumerate(treatments):
        if insideout:
            for location in [True, False]:
                data_l = data[data['inside injury'] == location]
                df = get_percentile_data(data_l, tx, i, controls, variables, quantiles)
                dfs.append(df)
        else:
            df = get_percentile_data(data, tx, i, controls, variables, quantiles)
            dfs.append(df)
    df = pd.concat(dfs).reset_index(drop=True)
    df.to_csv(save_path)
    quantiles = pd.DataFrame(quantiles)
    n = Path(save_path).stem + '_quantiles.csv'
    sp_q = os.path.join(Path(save_path).parents[0], n)
    quantiles.to_csv(sp_q)
    # Make plots
    fig, ax = plt.subplots(len(variables), 2)
    for i, v in enumerate(variables):
        mqdf  = quantiles[(quantiles['treatment'] == 'MIPS') & (quantiles['variable'] == v)]
        dqdf  = quantiles[(quantiles['treatment'] == 'DMSO (MIPS)') & (quantiles['variable'] == v)]
        ax0 = ax[i, 0]
        ax0.axline((0, 100), (1, 100), color='grey', alpha=0.5)
        y0 = f'{v} pcnt veh'
        sns.lineplot(data=df, x='Percentile', y=y0, ax=ax0, hue='treatment', marker='o', palette=pal1)
        ax1 = ax[i, 1]
        x1 = f'Control {v}'
        y1 = f'Treatment {v}'
        p0 = df[x1].min()
        p1 = df[y1].min()
        # grey line with gradient of 1
        ax1.axline((p0, p0), (p0 + 1, p0 + 1), color='grey', alpha=0.5)
        # Blue line at MIPS == 50%
        M50 = mqdf[mqdf['quantile'] == 50]['tx_val'].values[0]
        ax1.axline((p0, M50), (p0 + 1, M50), color=pal1['MIPS'], alpha=0.4, ls='--')
        # Light blue line at DMSO == 50%
        D50 = dqdf[dqdf['quantile'] == 50]['tx_val'].values[0]
        ax1.axline((D50, p1), (D50, p1 + 1), color=pal1['DMSO (MIPS)'], alpha=0.4, ls='--')
        # plot it 
        sns.lineplot(data=df, x=x1, y=y1, ax=ax1, hue='treatment', marker='o', palette=pal1)
    matplotlib.rcParams.update({'font.size': 10})
    #fig.subplots_adjust(right=0.95, left=0.125, bottom=0.074, top=0.96, wspace=0.485, hspace=0.337)
    fig.subplots_adjust(right=0.95, left=0.17, bottom=0.11, top=0.95, wspace=0.45, hspace=0.4)
    fig.set_size_inches(6, 8)
    plt.show()


def get_percentile_data(data, tx, i, controls, variables, quantiles):
    txdf = data[data['treatment'] == tx]
    ctrldf = data[data['treatment'] == controls[i]]
    #df = pd.concat([txdf, ctrldf])
    if len(txdf) < len(ctrldf):
        n = len(txdf)
    else:
        n = len(ctrldf)
    pcnt = np.linspace(0, 100, n)
    df = {}
    df['Percentile'] = pcnt
    df['treatment'] = [tx, ] * len(pcnt)
    for v in variables:
        tx_score = [stats.scoreatpercentile(txdf[v].values, p) for p in pcnt]
        ctrl_score = [stats.scoreatpercentile(ctrldf[v].values, p) for p in pcnt]
        df[f'Control {v}'] = ctrl_score
        df[f'Treatment {v}'] = tx_score
        add_quantiles(quantiles, txdf, tx, v)
        add_quantiles(quantiles, ctrldf, controls[i], v)
    df = pd.DataFrame(df)
    for v in variables:
        pcnt_veh_simp(v, df)
    return df


def add_quantiles(quantiles, txdf, tx, v):
    quants = [stats.scoreatpercentile(txdf[v].values, x) for x in [0, 25, 50, 100]]
    quantiles['tx_val'] = np.concatenate([quantiles['tx_val'], quants])
    quantiles['quantile'] = np.concatenate([quantiles['quantile'], [0, 25, 50, 100]])
    quantiles['treatment'] = np.concatenate([quantiles['treatment'], [tx, tx, tx, tx]])
    quantiles['variable'] = np.concatenate([quantiles['variable'], [v, v, v, v]])


@curry
def pcnt_veh(v, df):
    tx = df[f'Treatment {v}'].values
    ct = df[f'Control {v}'].values
    all = np.concatenate([tx, ct])
    all = np.expand_dims(all, axis=1)
    #min_val = all.min()
    #max_val = all.max()
    scaler = StandardScaler()
    scaler.fit(all)
    tx = np.expand_dims(tx, axis=1)
    tx = scaler.transform(tx)
    ct = np.expand_dims(ct, axis=1)
    ct = scaler.transform(ct)
    #ct_mean = np.squeeze(ct).mean()
    ct = np.squeeze(ct)
    tx = np.squeeze(tx)
    #tx = tx - min_val #) / (max_val - min_val)
    #ct = ct - min_val #) / (max_val) - min_val
    df[f'{v} pcnt veh'] = (tx / ct) * 100 #(tx / ct) * 100


def pcnt_veh_simp(v, df):
    tx = df[f'Treatment {v}'].values
    ct = df[f'Control {v}'].values
    #ct_mean = ct.mean()
    df[f'{v} pcnt veh'] = tx / ct * 100 #(tx / ct) * 100

def read_and_prep_data():
    d = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes'
    ns = ['211206_mips_df.parquet', '211206_veh-mips_df.parquet', 
                  '211206_cang_df.parquet', '211206_saline_df_220827_amp0.parquet', 
                  '211206_sq_df.parquet', '211206_veh-sq_df.parquet', '230301_MIPS_and_DMSO.parquet']
    ps = [os.path.join(d, n) for n in ns]
    df = []
    for p in ps:
        data = pd.read_parquet(p)
        if 'rho' not in data.columns.values:
            print('adding rho')
            data = spherical_coordinates(data)
            data.to_parquet(p)
        if 'cyl_r' not in data.columns.values:
            print('adding cyl_r')
            data = add_cylr(data)
            data.to_parquet(p)
        if 'time (s)' not in data.columns.values:
            print('adding time (s)')
            data = add_time_seconds(data)
            data.to_parquet(p)
        if 'outer_edge' not in data.columns.values:
            print('adding outer edge...')
            data = add_outeredge_var(data)
            data.to_parquet(p)
        df.append(data)
    del data
    df = pd.concat(df).reset_index(drop=True)
    df['treatment'] = df['path'].apply(get_treatment_name)
    return df


def inside_out(
        df, 
        peaks_data, 
        save_path,
        treatments=('MIPS', 'SQ', 'cangrelor'), 
        controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline'), ):
    data = {
        'path' : [], 
        'inside injury' : [], 
        'treatment' : [],
        'time from peak count' : [],
        'platelet count' : [], 
        'platelet density um^-3' : []
    }
    df = inside_injury_var(df)
    df = df[df['nrtracks'] > 1]
    ldf = df.head()
    ldf.to_csv(os.path.join(Path(save_path).parents[0], 'insideout_debugging.csv'))
    for i, ctl in enumerate(controls):
        tx = treatments[i]
        ttp = peaks_data[peaks_data['treatment'] == ctl]['time peak count'].mean()
        sdf = pd.concat([df[df['treatment'] == tx], df[df['treatment'] == ctl]])
        sdf['time from peak count'] = sdf['time (s)'] - ttp
        for k, grp in sdf.groupby(['path', 'time from peak count', 'treatment', 'inside_injury']):
            data['path'].append(k[0])
            data['time from peak count'].append(k[1])
            data['treatment'].append(k[2])
            data['inside injury'].append(k[3])
            data['platelet count'].append(len(pd.unique(grp.particle)))
            data['platelet density um^-3'].append(np.nanmean(grp['nb_density_15'].values))
    data = pd.DataFrame(data)
    data.to_csv(save_path)
    return data



def inside_out_size(
        df, 
        peaks_data, 
        save_path,
        treatments=('MIPS', 'SQ', 'cangrelor'), 
        controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline'), ):
    data = {
        'path' : [], 
        'inside injury' : [], 
        'size' : [],
        'treatment' : [],
        'time from peak count' : [],
        'platelet count' : [], 
        'platelet density um^-3' : []
    }
    df = size_var(df)
    df = inside_injury_var(df)
    df = df[df['nrtracks'] > 1]
    ldf = df.head()
    ldf.to_csv(os.path.join(Path(save_path).parents[0], 'insideout_debugging.csv'))
    for i, ctl in enumerate(controls):
        tx = treatments[i]
        ttp = peaks_data[peaks_data['treatment'] == ctl]['time peak count'].mean()
        sdf = pd.concat([df[df['treatment'] == tx], df[df['treatment'] == ctl]])
        sdf['time from peak count'] = sdf['time (s)'] - ttp
        for k, grp in sdf.groupby(['path', 'time from peak count', 'treatment', 'inside_injury', 'size']):
            data['path'].append(k[0])
            data['time from peak count'].append(k[1])
            data['treatment'].append(k[2])
            data['inside injury'].append(k[3])
            data['size'].append(k[4])
            data['platelet count'].append(len(pd.unique(grp.particle)))
            data['platelet density um^-3'].append(np.nanmean(grp['nb_density_15'].values))
    data = pd.DataFrame(data)
    data.to_csv(save_path)
    return data


if __name__ == '__main__':
    from datetime import datetime
    now = datetime.now()
    date = now.strftime("%y%m%d")
    #date = 230420
    #df = read_and_prep_data()
    #print(df.columns.values)
    insideout = True
    #if not insideout:
    #    nstr = ''
    #else:
    #    nstr = '_insideout'
    nstr = ''
    # NEW DATA
    df = read_and_prep_data()
    print(df.columns.values)

    # FROM FILE
    #sum_p = f'/Users/abigailmcgovern/Data/platelet-analysis/MIPS/Figure_3/{date}_count-and-growth-pcnt{nstr}_rolling-counts.csv'
    #summary_data = pd.read_csv(sum_p)
    #out_p = f'/Users/abigailmcgovern/Data/platelet-analysis/MIPS/Figure_3/{date}_count-and-growth-pcnt{nstr}_centile-data.csv'
    #out = pd.read_csv(out_p)
    #out_g_p = f'/Users/abigailmcgovern/Data/platelet-analysis/MIPS/Figure_3/{date}_count-and-growth-pcnt{nstr}_centile-data-growth.csv'
    #out_g = pd.read_csv(out_g_p)
    #out_c_p = f'/Users/abigailmcgovern/Data/platelet-analysis/MIPS/Figure_3/{date}_count-and-growth-pcnt{nstr}_centile-data-consolidation.csv'
    #out_c = pd.read_csv(out_c_p)


    #sns.lineplot(x='time (s)', y='platelet count', data=summary_data, hue='treatment')
    #plt.show()
    #sns.lineplot(x='time (s)', y='growth (platelets/s)', data=summary_data, hue='treatment')
    #plt.show()
    #pp_plot(out, ['max count', 'max growth rate (platelets/s)', 'count AUC', 'mean count'])
    #pp_plot(out, ['max size (um)', 'max growth (um/s)', 'size AUC (um^2)', 'mean size (um)'])
    #pp_plot(out, ['max density (platelets/um^2)', 'max contraction (platelets/um^2/s)',
     #             'density AUC (platelets/um^4)', 'mean density (platelets/um^2)'])
    #save_path_0 = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/percentile_analysis/centile-interpolated-data-count-3.csv'
    #save_path_1 = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/percentile_analysis/centile-interpolated-data-size-3.csv'
    #comparative_pp_plots(out, ['max count', 'max growth (platelets/s)', 'time peak count', 'mean count'], save_path_0)
    #comparative_pp_plots(out, ['max size (um)', 'max growth (um/s)', 'size AUC (um^2)', 'mean size (um)'], save_path_1)


    if not insideout:
        save_path = f'/Users/abigailmcgovern/Data/platelet-analysis/MIPS/Figure_3/{date}_count-and-growth-pcnt{nstr}.csv'
        out, out_g, out_c, summary_data, peaks_data = centile_score_data(df, save_path, insideout=insideout)
        save_path_g = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/percentile_analysis/centile-interpolated-data-count-growth.csv'
        comparative_pp_plots(out_g, ['max count', 'max growth (platelets/s)', 'time peak count', 'mean count'], save_path_g)
        save_path_c = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/percentile_analysis/centile-interpolated-data-count-consolidation.csv'
        comparative_pp_plots(out_c, ['max count', 'max growth (platelets/s)', 'time peak count', 'mean count'], save_path_c)
    else:
        date = 230420
        psp = f'/Users/abigailmcgovern/Data/platelet-analysis/MIPS/Figure_3/{date}_count-and-growth-pcnt_peaks.csv'
        peaks = pd.read_csv(psp)
        now = datetime.now()
        date = now.strftime("%y%m%d")
        #save_path = f'/Users/abigailmcgovern/Data/platelet-analysis/MIPS/Figure_3/{date}_inside_outside_counts_density.csv'
        #data = inside_out(df, peaks, save_path)

        save_path = f'/Users/abigailmcgovern/Data/platelet-analysis/MIPS/Figure_3/{date}_inside_outside_size_counts_density.csv'
        data = inside_out_size(df, peaks, save_path)
        ## GROWTH
        #save_path_g = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/percentile_analysis/centile-interpolated-data-count-growth_inside.csv'
        #comparative_pp_plots(in_out_g, ['max count', 'max growth (platelets/s)', 'time peak count', 'mean count'], save_path_g)
        #save_path_g = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/percentile_analysis/centile-interpolated-data-count-growth_outside.csv'
        #comparative_pp_plots(out_out_g, ['max count', 'max growth (platelets/s)', 'time peak count', 'mean count'], save_path_g)
        #
        ## CONSOLIDATION
        #save_path_c = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/percentile_analysis/centile-interpolated-data-count-consolidation_inside.csv'
        #comparative_pp_plots(in_out_c, ['max count', 'max growth (platelets/s)', 'time peak count', 'mean count'], save_path_c)
        #save_path_c = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/percentile_analysis/centile-interpolated-data-count-consolidation_outside.csv'
        #comparative_pp_plots(out_out_c, ['max count', 'max growth (platelets/s)', 'time peak count', 'mean count'], save_path_c)

        ## DEBUGGING
        #summary_data = summary_data[summary_data['treatment'] != 'DMSO (salgav)']
        #sns.lineplot(data=summary_data[summary_data['inside injury'] == True], y='platelet count', x='time (s)', hue='treatment', palette=pal1)
        #plt.show()
        #sns.lineplot(data=summary_data[summary_data['inside injury'] == False], y='platelet count', x='time (s)', hue='treatment', palette=pal1)
        #plt.show()


