import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy import stats
from collections import defaultdict
from plateletanalysis.topology.donutness import summary_donutness
from dtaidistance import dtw
from plateletanalysis.analysis.summary_measures import recruitment_phase, shedding_phase, \
    platelet_mean_of_var, p_recruited_gt60, p_recruited_lt15, p_gt60s, p_lt15s, \
        initial_platelet_velocity_change, var_for_first_3_frames
import seaborn as sns
import matplotlib.pyplot as plt 
from plateletanalysis.variables.basic import psel_bin


# -------------------
# Extract growth data
# -------------------

def growth_data(df):
    result = defaultdict(list)
    gb = ['path', 'time (s)']
    for k, grp in df.groupby(gb):
        for i, g in enumerate(gb):
            result[g].append(k[i])
        result['platelet count'].append(count(grp))
        result['average density'].append(platelet_mean('nb_density_15', grp))
        result['density SD'].append(platelet_std('nb_density_15', grp))
        result['average speed'].append(platelet_mean('dv', grp))
        result['speed SD'].append(platelet_std('dv', grp))
        result['average y-axis velocity'].append(platelet_mean('dvy', grp))
        result['average z-axis velocity'].append(platelet_std('dvy', grp))
        result['y-axis velocity SD'].append(platelet_mean('dvz', grp))
        result['z-axis velocity SD'].append(platelet_std('dvz', grp))
    result = pd.DataFrame(result)
    result = result.sort_values(by='time (s)')
    result = smooth_vars(result, ['platelet count', ], gb='path', w=30, add_suff=' smoothed (30)')
    for k, grp in result.groupby('path'):
        vals = np.diff(grp['platelet count'].values)
        vals = np.concatenate([np.array([np.NaN, ]), vals])
        idxs = grp.index.values
        result.loc[idxs, 'net growth'] = vals
        vals2 = np.diff(grp['platelet count smoothed (30)'].values)
        vals2 = np.concatenate([np.array([np.NaN, ]), vals2])
        result.loc[idxs, 'net growth smoothed'] = vals2
    result['net growth (%)'] = result['net growth'] / result['platelet count'] * 100
    result['net loss (%)'] = - result['net growth (%)']
    return result


def count(grp):
    grps = grp[grp['nrtracks'] > 1]
    return len(pd.unique(grps['particle']))


def platelet_mean(var, grp):
    platelet_means = grp.groupby('particle')[var].mean()
    return np.nanmean(platelet_means.values)

def platelet_std(var, grp):
    platelet_means = grp.groupby('particle')[var].std()
    return np.nanstd(platelet_means.values)

# --------------
# Peaks analysis
# --------------

def experiment_data_df(df, gdf, ddf, save_path, extra=False):
    results = defaultdict(list)
    gdf = smooth_vars(gdf, ('net growth smoothed', ), gb='path', w=15)  # extra smoothed
    gdf = smooth_vars(gdf, ('platelet count', 'average density', 'average speed'), w=15, gb='path', add_suff=' smoothed (15)')
    gdf = gdf.sort_values(by='time (s)')
    gdf['net growth (%)'] = gdf['net growth'] / gdf['platelet count'] * 100
    gdf['net loss (%)'] = - gdf['net growth (%)']
    ddf = summary_donutness(ddf)
    ddf = smooth_vars(ddf, ('donutness', ), gb='path', w=25)
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


# ----------------
# Dist from centre
# ----------------

def var_over_cylr(df, col, gb, gb1=['path', 'time_bin']):
    cols = [col, ]
    data = groupby_summary_data_mean(df, gb, cols)
    data = smooth_vars(data, vars=cols, gb=gb1, t='cylr_bin', w=4)
    data = data.dropna()
    return data


def plot_var_over_cylr(df, col):
    gb = ['path', 'time_bin', 'cylr_bin']
    data = var_over_cylr(df, col, gb)
    plt.rcParams['svg.fonttype'] = 'none'
    ax = sns.lineplot(data=data, y=col, x='cylr_bin', palette='rocket', hue='time_bin')
    ax.axline((37.5, 0), (37.5, 0.001), color='grey')
    sns.despine()
    plt.show()



def plot_var_over_cylr_ind(df, col):
    gb = ['path', 'cylr_bin']
    data = var_over_cylr(df, col, gb)
    plt.rcParams['svg.fonttype'] = 'none'
    ax = sns.lineplot(data=data, y=col, x='cylr_bin', palette='rocket', hue='path')
    ax.axline((37.5, 0), (37.5, 0.001), color='grey')
    sns.despine()
    plt.show()



def cylr_zerocrossing_box(df, col):
    gb = ['path', 'cylr_bin']
    cols = [col, ]
    df = df[df['time_bin'] == '0-60 s']
    data = groupby_summary_data_mean(df, gb, cols)
    data = smooth_vars(df, vars=cols, gb=['path', ], t='cylr_bin', w=8)
    data = mark_as_bookends(data, gb, col)
    data = data[data['bookends'] == True]
    data = data.groupby(['path', ])['cylr_bin'].mean().reset_index()
    data = data.dropna()
    print(data)
    print(len(data))
    print(len(pd.unique(data.path)))
    plt.rcParams['svg.fonttype'] = 'none'
    ax = sns.boxplot(data=data, y='cylr_bin', palette='rocket')
    sns.stripplot(data=data, y='cylr_bin', ax=ax, palette='rocket', edgecolor='white', linewidth=0.3, jitter=True, size=5,)
    sns.despine()
    plt.show()


def cylr_zerocrossing_box_mins(df, col):
    gb = ['path', 'time_bin', 'cylr_bin']
    cols = [col, ]
    df = df[(df['time_bin'] == '0-60 s') | (df['time_bin'] == '60-120 s')]
    data = groupby_summary_data_mean(df, gb, cols)
    data = smooth_vars(data, vars=cols, gb=['path', 'time_bin'], t='cylr_bin', w=8)
    data = mark_as_bookends(data, gb, col)
    data = data[data['bookends'] == True]
    data = data.groupby(['path', 'time_bin'])['cylr_bin'].mean().reset_index()
    data = data.dropna()
    plt.rcParams['svg.fonttype'] = 'none'
    ax = sns.boxplot(data=data, y='cylr_bin', x='time_bin', palette='rocket')
    sns.stripplot(data=data, y='cylr_bin',  x='time_bin', ax=ax, palette='rocket', edgecolor='white', linewidth=0.3, jitter=True, size=5,)
    sns.despine()
    plt.show()


def mark_as_bookends(data, gb, col):
    data = data.sort_values('cylr_bin')
    for k, grp in data.groupby(gb):
        prev = None
        vals = []
        idx = grp.index.values
        for i, v in enumerate(grp[col].values):
            if prev is not None and 0 > prev and 0 < v:
                vals.append(True)
            else:
                vals.append(False)
            prev = v
        data.loc[idx, 'bookends'] = vals
    return data



# -------
# Helpers
# -------

def smooth_vars(df, vars, w=15, t='time (s)', gb=['path', 'particle'], add_suff=None):
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


def groupby_summary_data_mean(df, gb, cols):
    data = defaultdict(list)
    for k, grp in df.groupby(gb):
        for i, c in enumerate(gb):
            data[c].append(k[i])
        for c in cols:
            data[c].append(np.nanmean(grp[c].values))
    data = pd.DataFrame(data)
    return data



def groupby_summary_data_counts(df, gb):
    data = defaultdict(list)
    for k, grp in df.groupby(gb):
        for i, c in enumerate(gb):
            data[c].append(k[i])
        data['platelet count'].append(len(pd.unique(grp['particle'])))
    data = pd.DataFrame(data)
    return data


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


# ----------
# P selectin
# ----------
    
def psel_pcnt(df):
    n = len(pd.unique(df['particle']))
    psel = df[df['psel'] == True]
    p = len(pd.unique(psel['particle']))
    return p / n * 100
    

def percent_psel_pos(df, gb=['path', 'time (s)']):
    if 'psel' not in df.columns.values:
        df = psel_bin(df)
    data = df.groupby(gb).apply(psel_pcnt).reset_index()
    data = data.rename(columns={0 :  'p-selectin positive platelets (%)'})
    return data


