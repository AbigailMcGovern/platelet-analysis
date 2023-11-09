import pandas as pd
import numpy as np
from scipy.signal import find_peaks, peak_widths, peak_prominences
from scipy.fft import fft, fftfreq
from scipy import stats
from collections import defaultdict
import os
import seaborn as sns
import matplotlib.pyplot as plt
from plateletanalysis.variables.basic import time_seconds, get_treatment_name, add_terminating, time_tracked_var
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
from toolz import curry
from plateletanalysis.analysis.summary_measures import recruitment_phase, shedding_phase, \
    platelet_mean_of_var, p_recruited_gt60, p_recruited_lt15, p_gt60s, p_lt15s, \
        initial_platelet_velocity_change, var_for_first_3_frames


# Basic
# -----

def smooth_vars(df, vars, w=20, t='time (s)', gb='path', add_suff=None):
    df = df.sort_values(t)
    for v in vars:
        if add_suff is not None:
            v_n = v + add_suff
        else:
            v_n = v
        for k, grp in df.groupby(gb):
            rolled = grp[v].rolling(window=w, center=True).mean()
            idxs = grp.index.values
            df.loc[idxs, v_n] = rolled
    return df


def classify_exp_type(path):
    if path.find('exp5') != -1:
        return '10-20 min'
    elif path.find('exp3') != -1:
        return '0-10 min'
    else:
        return 'other'


# experiment wise
# ---------------
def experiment_data_df(df, gdf, ddf, save_path, extra=False):
    results = defaultdict(list)
    gdf = smooth_vars(gdf, ('net growth smoothed', ), w=15)  # extra smoothed
    gdf = smooth_vars(gdf, ('platelet count', 'average density', 'average speed'), w=15, add_suff=' smoothed (15)')
    gdf = gdf.sort_values(by='time (s)')
    gdf['net growth (%)'] = gdf['net growth'] / gdf['platelet count'] * 100
    gdf['net loss (%)'] = - gdf['net growth (%)']
    ddf = summary_donutness(ddf)
    ddf = smooth_vars(ddf, ('donutness', ), w=25)
    for k, grp in gdf.groupby('path'):
        if k != 'average':
            # prep
            t_max = grp['time (s)'].max()
            ddf_grp = ddf[ddf['path'] == k]
            results['path'].append(k)
            df_grp = df[df['path'] == k]
            # vals 
            max_count = grp['platelet count'].max()
            results['max platelet count'].append(max_count)
            results['mean platelet count'].append(grp['platelet count'].mean())
            # dynamic time warping
            count_dist = dtw_difference(grp, gdf, 'platelet count')
            results['DTW count dist'].append(count_dist)
            growth_dist = dtw_difference(grp, gdf, 'net growth smoothed')
            results['DTW growth dist'].append(growth_dist)
            # latencies to maximum
            count_tmax = time_to_max(grp, 'platelet count smoothed (15)')
            results['latency max count'].append(count_tmax)
            growth_tmax = time_to_max(grp, 'net growth smoothed')
            results['latency max growth'].append(growth_tmax)
            # some subdivisions for summary
            growth = grp[grp['time (s)'] <= count_tmax]
            results['mean platelet count -G'].append(growth['platelet count'].mean())
            consol = grp[grp['time (s)'] > count_tmax]
            results['mean platelet count -C'].append(consol['platelet count'].mean())
            # subdivisions for full df
            growth_df = df_grp[df_grp['time (s)'] <= count_tmax]
            consol_df = df_grp[df_grp['time (s)'] > count_tmax]
            # number, time between, % of count, and frequency of embolysis events
            emb_n, emb_ip, emb_h, emb_f = peak_analysis_frequency(grp, 'net loss (%)', 
                                                                  height=10, t_max=t_max, t_col='time (s)')
            results['n embolysis events'].append(emb_n)
            results['average time between embolysis events (s)'].append(emb_ip)
            results['average size embolysis events (%)'].append(emb_h)
            results['frequency embolysis events (/s)'].append(emb_f)
            emb_n_g, emb_ip_g, emb_h_g, emb_f_g = peak_analysis_frequency(growth, 'net loss (%)', 
                                                                          height=10, t_max=t_max, t_col='time (s)')
            results['n embolysis events - G'].append(emb_n_g)
            results['average time between embolysis events (s) - G'].append(emb_ip_g)
            results['average size embolysis events (%) - G'].append(emb_h_g)
            results['frequency embolysis events (/s) - G'].append(emb_f_g)
            emb_n_c, emb_ip_c, emb_h_c, emb_f_c = peak_analysis_frequency(consol, 'net loss (%)',
                                                                          height=10, t_max=t_max, t_col='time (s)')
            results['n embolysis events - C'].append(emb_n_c)
            results['average time between embolysis events (s) - C'].append(emb_ip_c)
            results['average size embolysis events (%) - C'].append(emb_h_c)
            results['frequency embolysis events (/s) - C'].append(emb_f_c)
            # Donutness peak analysis
            duration, height, prominence, mean, latency = peak_analysis_size(ddf_grp, 'donutness', height=1.5, 
                                                                    dist=300, t_col='time (s)', 
                                                                    prom=0.00001, first_only=True)
            results['donutness duration (s)'].append(duration)
            results['donutness magnetude'].append(height)
            results['donutness prominence'].append(prominence)
            results['donutness mean'].append(mean)
            results['latency donutness'].append(latency)
            # recruitment and shedding
            recruitment = recruitment_phase(df_grp)
            results['recruitment'].append(recruitment)
            recruitment_g = recruitment_phase(growth_df)
            results['recruitment - G'].append(recruitment_g)
            recruitment_c = recruitment_phase(consol_df)
            results['recruitment - C'].append(recruitment_c)
            shedding = shedding_phase(df_grp)
            results['shedding'].append(shedding)
            shedding_g = shedding_phase(growth_df)
            results['shedding - G'].append(shedding_g)
            shedding_c = shedding_phase(consol_df)
            results['shedding - C'].append(shedding_c)
            prec_15 = p_recruited_lt15(df_grp)
            results['P(recruited < 15 s)'].append(prec_15)
            prec_15_g = p_recruited_lt15(growth_df)
            results['P(recruited < 15 s) - G'].append(prec_15_g)
            prec_15_c = p_recruited_lt15(consol_df)
            results['P(recruited < 15 s) - C'].append(prec_15_c)
            prec_60 = p_recruited_gt60(df_grp)
            results['P(recruited > 60 s)'].append(prec_60)
            prec_60_g = p_recruited_gt60(growth_df)
            results['P(recruited > 60 s) - G'].append(prec_60_g)
            prec_60_c = p_recruited_gt60(consol_df)
            results['P(recruited > 60 s) - C'].append(prec_60_c)
            if extra:
                # props
                plt15 = p_lt15s(df_grp)
                results['P(< 15 s)'].append(plt15)
                plt15_g = p_lt15s(growth_df)
                results['P(< 15 s) - G'].append(plt15_g)
                plt15_c = p_lt15s(consol_df)
                results['P(< 15 s) - C'].append(plt15_c)
                pgt60 = p_gt60s(df_grp)
                results['P(> 60 s)'].append(pgt60)
                pgt60_g = p_gt60s(growth_df)
                results['P(> 60 s) - G'].append(pgt60_g)
                pgt60_c = p_gt60s(consol_df)
                results['P(> 60 s) - C'].append(pgt60_c)
                # velocity
                dv = platelet_mean_of_var('dv', df_grp)
                results['dv'].append(dv)
                dvy = platelet_mean_of_var('dvy', df_grp)
                results['dvy'].append(dvy)
                dvz = platelet_mean_of_var('dvz', df_grp)
                results['dvz'].append(dvz)
                dv_g = platelet_mean_of_var('dv', growth_df)
                results['dv - G'].append(dv_g)
                dvy_g = platelet_mean_of_var('dvy', growth_df)
                results['dvy - G'].append(dvy_g)
                dvz_g = platelet_mean_of_var('dvz', growth_df)
                results['dvz - G'].append(dvz_g)
                dv_c = platelet_mean_of_var('dv', consol_df)
                results['dv - C'].append(dv_c)
                dvy_c = platelet_mean_of_var('dvy', consol_df)
                results['dvy - C'].append(dvy_c)
                dvz_c = platelet_mean_of_var('dvz', consol_df)
                results['dvz - C'].append(dvz_c)
                # Density
                dens = platelet_mean_of_var('nb_density_15', df_grp)
                results['density'].append(dens)
                dens_g = platelet_mean_of_var('nb_density_15', growth_df)
                results['density - G'].append(dens_g)
                dens_c = platelet_mean_of_var('nb_density_15', consol_df)
                results['density - C'].append(dens_c)
                # nrtracks
                nrt = platelet_mean_of_var('nrtracks', df_grp)
                results['nrtracks'].append(nrt)
                nrt_g = platelet_mean_of_var('nrtracks', growth_df)
                results['nrtracks - G'].append(nrt_g)
                nrt_c = platelet_mean_of_var('nrtracks', consol_df)
                results['nrtracks - C'].append(nrt_c)
                # stability
                stab = platelet_mean_of_var('stab', df_grp)
                results['instability'].append(stab)
                stab_g = platelet_mean_of_var('stab', growth_df)
                results['instability - G'].append(stab_g)
                stab_c = platelet_mean_of_var('stab', consol_df)
                results['instability - C'].append(stab_c)
                # decelleration 
                decel = - initial_platelet_velocity_change(df_grp)
                results['initial decelleration'].append(decel)
                decel_g = - initial_platelet_velocity_change(growth_df)
                results['initial decelleration - G'].append(decel_g)
                decel_c = - initial_platelet_velocity_change(consol_df)
                results['initial decelleration - C'].append(decel_c)
                # initial density 
                idens = var_for_first_3_frames('nb_density_15', df_grp)
                results['initial density'].append(idens)
                idens_g = var_for_first_3_frames('nb_density_15', growth_df)
                results['initial density - G'].append(idens_g)
                idens_c = var_for_first_3_frames('nb_density_15', consol_df)
                results['initial density - C'].append(idens_c)
                # initial stability
                istab = var_for_first_3_frames('stab', df_grp)
                results['initial instability'].append(istab)
                istab_g = var_for_first_3_frames('stab', growth_df)
                results['initial instability - G'].append(istab_g)
                istab_c = var_for_first_3_frames('stab', consol_df)
                results['initial instability - C'].append(istab_c)

    results = pd.DataFrame(results)
    results.to_csv(save_path)
    return results



# Per-thrombus measurements
# -------------------------

def dtw_difference(grp, df, col):
    # assume sorted by time
    vals = grp[col].values
    av = df.groupby('time (s)')[col].apply(np.mean).values
    if len(vals) != len(av):
        diff = len(av) - len(vals)
        av = av[:-diff]
    dist = dtw.distance(vals, av)
    return dist


def time_to_max(grp, var, time_col='time (s)'):
    vals = grp[var].values
    idx = np.nanargmax(vals)
    ts = grp[time_col].values
    t = ts[idx]
    return t


def peak_analysis_frequency(grp, var, t_max, t_col='time (s)', height=None, prom=None, dist=None):
    vals = grp[var].values
    times = grp[t_col].values
    peaks, prop = find_peaks(vals, height=height, distance=dist, prominence=prom)
    ts = times[peaks]
    heights = vals[peaks]
    n_peaks = len(ts)
    average_interpeak = np.nanmean(np.diff(ts))
    average_height = np.mean(heights)
    frequency = n_peaks / t_max
    return n_peaks, average_interpeak, average_height, frequency


def peak_analysis_size(grp, var, t_col='time (s)', height=None, prom=None, dist=None, first_only=True):
    vals = grp[var].values
    times = grp[t_col].values
    peaks, prop = find_peaks(vals, height=height, distance=dist, prominence=prom)
    p = prop['prominences']
    lb = prop['left_bases']
    rb = prop['right_bases']
    h = prop['peak_heights']
    if len(p) == 0:
        duration = 0
        height = np.nanmean(vals)
        prominence = 0
        mean = np.nanmean(vals)
        latency = 0
    elif len(p) == 1 or first_only:
        duration = times[rb[0]] - times[lb[0]]
        ar = vals[lb[0]:rb[0]]
        mean = np.nanmean(ar)
        height = h[0]
        prominence = p[0]
        latency = times[peaks[0]]
    else:
        duration = np.nanmean([times[rb[i]] - times[lb[i]] for i in range(len(p))])
        mean = np.nanmean([vals[lb[i]:rb[i]].mean() for i in range(len(p))])
        height = np.nanmean(h)
        prominence = np.nanmean(p)
        latency = np.nanmean([times[peaks[i]] for i in range(len(p))])

    return duration, height, prominence, mean, latency

# experiment x time point
# -----------------------

def exp_by_time_donutness_df(donutness_dfs, filter_types, data_df):
    result = growth_data(data_df)
    result = smooth_vars(result, ('platelet count', ), w=20)
    result = result.set_index(['path', 'time'], drop = True)
    for df, n in zip(donutness_dfs, filter_types):
        df = smooth_vars(df, ('donutness', ), w=20)
        for k, grp in df.groupby(['path', 'time (s)']):
            result.loc[k, 'donutness']


def time_above_sim_stats(df, sim_df, result):
    result = result.set_index(['path', 'time'], drop = True)
    n_tp = len(pd.unique(df.path))
    for k, grp in df.groupby(['path', 'time (s)']):
        data = grp['donutness'].values
        sim_data = sim_df[sim_df['time (s)'] == k[1]]['donutness'].values
        t, p = stats.ttest_ind(data, sim_data, alternative='greater')
        p = p / n_tp # bonforroni
        result.loc[k, 'donutness_gt_sim_t'] = t
        result.loc[k, 'donutness_gt_sim_p'] = p
    result = result.reset_index()
    return result


def growth_data(df, columns_dict):
    result = defaultdict(list)
    gb = ['path', 'time (s)']
    for k, grp in df.groupby(gb):
        for i, g in enumerate(gb):
            result[g].append(k[i])
            result['platelet count'].append(count(grp))
            result['average density'].append(grp['nb_density_15'].mean())
            result['density SD'].append(grp['nb_density_15'].std())
            result['average speed'].append(grp['dv'].mean())
            result['speed SD'].append(grp['dv'].std())
            result['average y-axis velocity'].append(grp['dvy'].mean())
            result['average z-axis velocity'].append(grp['dvz'].mean())
            result['y-axis velocity SD'].append(grp['dvy'].std())
            result['z-axis velocity SD'].append(grp['dvz'].std())
            for c in columns_dict.keys():
                result[c].append(grp[columns_dict[c]].mean())
                sdn = c + ' SD'
                result[sdn].append(grp[columns_dict[c]].std())
    result = pd.DataFrame(result)
    result = result.sort_values(by='time (s)')
    result = smooth_vars(result, ['platelet count', ], w=30, add_suff=' smoothed (30)')
    for k, grp in result.groupby('path'):
        vals = np.diff(grp['platelet count'].values)
        vals = np.concatenate([np.array([np.NaN, ]), vals])
        idxs = grp.index.values
        result.loc[idxs, 'net growth'] = vals
        vals2 = np.diff(grp['platelet count smoothed (30)'].values)
        vals2 = np.concatenate([np.array([np.NaN, ]), vals2])
        result.loc[idxs, 'net growth smoothed'] = vals2
    return result


def count(grp):
    return len(pd.unique(grp['particle']))



# For Donutness
# -------------

def average_donutness_vs_count(ddf, cdf):
    result = defaultdict(list)
    for k, grp in ddf.groupby('path'):
        result['path'].append(k)
        result['donutness'].append(grp['donutness'].mean())
        result['count'].append(cdf[cdf['path'] == k]['platelet count'].mean())
    result = pd.DataFrame(result)
    sns.scatterplot(data=result, x='donutness', y='count', hue='path')
    plt.show()


def summary_donutness(ddf):
    res = defaultdict(list)
    for k, grp in ddf.groupby(['path', 'time (s)']):
        res['path'].append(k[0])
        res['time (s)'].append(k[1])
        res['donutness'].append(grp['donutness'].mean())
    res = pd.DataFrame(res)
    return res



# Curve experimentation
# ---------------------

def find_curves_exp(df, var, w=None, height=None, prom=None, dist=None, t_col='time (s)'):
    if w is not None:
        df = smooth_vars(df, (var, ), w=w)
    df = df.sort_values(by=t_col)
    result = defaultdict(list)
    # grp
    for k, grp in df.groupby('path'):
        vals = grp[var].values
        times = grp[t_col].values
        peaks, prop = find_peaks(vals, height=height, distance=dist, prominence=prom)
        print(k)
        print(prop)
        ts = times[peaks]
        heights = vals[peaks]
        result['path'] = result['path'] + [k, ] * len(ts)
        result['time (s)'] = result['time (s)'] + list(ts)
        result[var] = result[var] + list(heights)
    result = pd.DataFrame(result)
    fig, ax = plt.subplots(1, 1)
    sns.lineplot(data=df, x='time (s)', y=var, ax=ax, hue='path')
    sns.scatterplot(data=result, x='time (s)', y=var, ax=ax, hue='path')
    plt.show()



def find_frequencies_exp(df, var):
    df = df.sort_values(by='time (s)')
    result = defaultdict(list)
    paths = pd.unique(df['path'])
    N = len(pd.unique(df['time (s)']))
    sdf = df[df['path'] == paths[0]]
    T = sdf['time (s)'].values[1] - sdf['time (s)'].values[0]
    xf = fftfreq(N, T)[:N//2]
    for k, grp in df.groupby('path'):
        y = - grp[var].values
        y = np.nan_to_num(y)
        yf = fft(y)
        yf = 2.0/N * np.abs(yf[0:N//2])
        result['frequency (/s)'] = result['frequency (/s)'] + list(xf)
        result['power'] = result['power'] + list(yf)
        result['path'] = result['path'] + [k, ] * len(yf)
    result = pd.DataFrame(result)
    result = smooth_vars(result, ('power', ), w=20, t='frequency (/s)')
    return result
    #plt.show()


# -------
# Compute
# -------

if __name__ == '__main__':
    p = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/230919_p-selectin.parquet'
    sd = '/Users/abigailmcgovern/Data/platelet-analysis/toroidal_paper/figure_2'
    psel_chan = 'GaAsP Alexa 568: mean_intensity'
    cal_chan = 'GaAsP Alexa 488: mean_intensity'

    # Read data
    # ---------
    df = pd.read_parquet(p)
    #rename = {psel_chan : 'p-sel average intensity', 
    #          cal_chan : 'calcium average intensity'}
    #df = df.rename(columns=rename)
    #df = time_seconds(df)



    # Growth and summary data
    # -----------------------
    n = '230930_p-sel_growth_and_summary_data.csv'
    columns_dict = {
        'P-selectin fluorescence' : 'p-sel average intensity', 
        'Calcium fluorescence' : 'calcium average intensity',}
    sp0 = os.path.join(sd, n)
    #out = growth_data(df, columns_dict)
    #out.to_csv(sp0)
    out = pd.read_csv(sp0)
    out['treatment'] = out['path'].apply(get_treatment_name)
    out = out[out['treatment'] == 'control']
    out['exp_type'] = out['path'].apply(classify_exp_type)
    out = out[out['exp_type'] == '0-10 min']
    #out = smooth_vars(out, ('platelet count', 'average density', 'average speed'), w=10)
    #out = smooth_vars(out, ('net growth smoothed', ), w=15)    
    #sns.lineplot(data=out, x='time (s)', y='platelet count', hue='path')
    #plt.show()
    #sns.lineplot(data=out, x='time (s)', y='average density', hue='path')
    #plt.show()
    #sns.lineplot(data=out, x='time (s)', y='average speed', hue='path')
    #plt.show()



    # Donutness
    # ---------
    sp1 = os.path.join(sd, 'p-sel_density_donutness_data.csv')
    ddf = pd.read_csv(sp1)
    ddf = time_seconds(ddf)
    ddf['treatment'] = ddf['path'].apply(get_treatment_name)
    ddf = ddf[ddf['treatment'] == 'control']
    ddf['exp_type'] = ddf['path'].apply(classify_exp_type)
    ddf = ddf[ddf['exp_type'] == '0-10 min']
    print('n control = ', len(pd.unique(ddf.path))) # 11

    #print(pd.unique(ddf.treatment))
    #ddf = ddf[ddf['treatment'] == 'DMSO (MIPS)']
    #print('n DMSO (MIPS) = ', len(pd.unique(ddf.path))) # 5
    print(pd.unique(ddf.path))
    # 220526_IVMTR143_Inj4_Ctrl_exp3
    # 220526_IVMTR143_Inj5_Ctrl_exp3 - missing data
    ddf = ddf[ddf['path'] != '220526_IVMTR143_Inj4_Ctrl_exp3']
    ddf = ddf[ddf['path'] != '220526_IVMTR143_Inj5_Ctrl_exp3']
    out = out[out['path'] != '220526_IVMTR143_Inj4_Ctrl_exp3']
    out = out[out['path'] != '220526_IVMTR143_Inj5_Ctrl_exp3']
    #ddf = smooth_vars(ddf, ('donutness', ), gb=['path', 'bootstrap_id'])
    out['net growth (%)'] = out['net growth'] / out['platelet count'] * 100
    out['net loss (%)'] = - out['net growth (%)']
    #sns.lineplot(data=out, x='time (s)', y='net growth (%)', hue='path')
    #plt.show()
    #sns.lineplot(data=ddf, x='time (s)', y='donutness', hue='path')
    #plt.show() 
    #average_donutness_vs_count(ddf, out)



    # Peak measurements
    # -----------------
    #result = find_frequencies_exp(out, 'net growth (%)')
    #sns.lineplot(data=result, x='frequency (/s)', y='power', hue='path')
    #plt.show()

    #find_curves_exp(out, 'net loss (%)', w=None, height=10, t_col='time (s)')

    #ddf_sum = summary_donutness(ddf)
    #find_curves_exp(ddf_sum, 'donutness', w=25, height=1.5, dist=300, t_col='time (s)', prom=0.00001)
    save_path = '/Users/abigailmcgovern/Data/platelet-analysis/P-selectin/growth_donutness_exp_summary_more-vars.csv'
    df = add_terminating(df)
    df = time_tracked_var(df)
    sum_df = experiment_data_df(df, out, ddf, save_path, extra=True)
    #sns.pairplot(sum_df)
    #plt.show()



