from ripser import ripser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import seaborn as sns
from pathlib import Path
from scipy import stats
from plateletanalysis.variables.measure import quantile_normalise_variables, quantile_normalise_variables_frame
from plateletanalysis.variables.position import scale_free_positional_categories, count_variables, transitions
from scipy.signal import find_peaks
from plateletanalysis.variables.neighbours import add_neighbour_lists, local_density
from sklearn.preprocessing import StandardScaler



# -----------------------------------------------------------------------------
# ------------------------------
# Major data aquisition function 
# ------------------------------
# -----------------------------------------------------------------------------

def donutness_data(
        df : pd.DataFrame, 
        sample_col='path', 
        time_col='frame',
        sample_n=200,
        n_samples=100,
        centile=50,
        filter_col='nb_density_15_pcntf', 
        y_col='ys', 
        x_col='x_s', 
        #get_accessory_data=False, 
        units='%'
    ):
    # only the upper centiles of the data (e.g., density, fibrin, p selectin)
    df = df[df[filter_col] > centile]
    df = scale_x_and_y(df)
    y_col_s = y_col + '_scaled'
    x_col_s = x_col + '_scaled'
    data_cols = [sample_col, x_col_s, y_col_s]
    if time_col is not None:
        data_cols.append(time_col)
    df = df[data_cols]
    samples = pd.unique(data[sample_col])
    ph_data = initialise_PH_data_dict()
    samples = pd.unique(df[sample_col])
    #tx_name = get_treatment_name(data['path'].values[0])
    #desc=f'Getting max barcode data for treatment = {tx_name}'
    n_its = len(samples) * n_samples
    with tqdm(total=n_its) as progress:
        for sample, data in df.groupby([sample_col, ]):
            for bs_id in range(n_samples):
                data = data.sample(n=sample_n)
                sample_persistent_homology_analysis(data, x_col_s, y_col_s, 
                                                    ph_data, sample_col, 
                                                    time_col, sample, bs_id)
                progress.update(1)
    ph_data = pd.DataFrame(ph_data)
    if time_col is not None:
        ph_data['time (s)'] = ph_data[time_col]
    #if get_accessory_data:
     #   _, donut_info = find_max_donut_time(out)
      #  out = accessory_platelet_data(out, df, donut_info)
    return ph_data



def initialise_PH_data_dict(sample_col, time_col):
    ph_data = {
        'boottrap_id' : [],
        'birth_1' : [],
        'birth_2' : [],
        'birth_mean' : [],
        'birth_std' : [],
        'death_1' : [], 
        'death_2' : [], 
        'death_mean' : [],
        'death_std' : [],
        'persistence_1' : [],
        'persistence_2' : [], 
        'persistence_mean' : [],
        'persistence_std' : [],
        'outlierness_1' : [], 
        'outlierness_2' : [],
        'outlierness_mean' : [],
        'outlierness_std' : [],
        'donutness' : []
    }
    ph_data[sample_col] = []
    if time_col is not None:
        ph_data[time_col] = []
    return ph_data



def sample_persistent_homology_analysis(
        data, 
        x_col, 
        y_col, 
        ph_data, 
        sample_col, 
        time_col, 
        sample, 
        bootstrap_id
    ):
    if time_col is not None:
        frames = list(range(data[time_col].max()))
    else:
        frames = [0, ]
    for t in frames:
        if time_col is not None:
            data_t = data[data['frame'] == t]
            ph_data[time_col].append(t)
        else:
            data_t = data
        X = data_t[[x_col, y_col]].values
        ph_data[sample_col].append(sample)
        ph_data['bootstrap_id'].append(bootstrap_id)
        if len(X) > 0:
            dgms = ripser(X)['dgms']
            h1 = dgms[1]
            #print(h1)
            if len(h1) > 1:
                births = h1[:, 0]
                deaths = h1[:, 1]
                persistence = deaths - births
                idxs = np.argsort(persistence)
                # births
                ph_data['birth_1'].append( births[idxs[-1]])
                ph_data['birth_s'].append(births[idxs[-2]])
                ph_data['birth_mean'].append(np.mean(births))
                ph_data['birth_std'].append(np.std(births))
                # deaths
                ph_data['death_1'].append(deaths[idxs[-1]])
                ph_data['death_2'].append(deaths[idxs[-2]])
                ph_data['death_mean'].append(np.mean(deaths))
                ph_data['death_std'].append(np.std(deaths))
                # persistence
                ph_data['persistence_1'].append(persistence[idxs[-1]])
                ph_data['persistence_2'].append(persistence[idxs[-1]])
                persistence_mean = np.mean(persistence)
                ph_data['persistence_mean'].append(persistence_mean)
                persistence_std = np.std(persistence)
                ph_data['persistence_std'].append(persistence_std)
                # outlierness
                outlierness = (persistence - persistence_mean) / persistence_std
                outlierness_1 = outlierness[idxs[-1]]
                ph_data['outlierness_1'].append(outlierness_1)
                outlierness_2 = outlierness[idxs[-2]]
                ph_data['outlierness_2'].append(outlierness_2)
                ph_data['outlierness_mean'].append(np.mean(outlierness))
                ph_data['outlierness_std'].append(np.std(outlierness))
                # donutness
                ph_data['donutness'] = outlierness_1 - outlierness_2
            else:
                append_NaN(ph_data, sample_col, time_col)
        else:
            append_NaN(ph_data, sample_col, time_col)
    
    return ph_data



def append_NaN(ph_data, sample_col, time_col):
    for k in ph_data.keys():
        if k != sample_col and k != time_col and k != 'bootstrap_id':
            ph_data[k].append(np.NaN)




# ---------------
# Additional Data
# ---------------

#TODO: Please change this function to add mean data to existing dataframe
# -i.e., move this to another module and make usable for any summary table:
#           - with path + frame
#           - for any categorical/condition 
#           (e.g., exp @ frame 100, only surface platelets) 

def accessory_platelet_data(
        out, 
        df, 
        donut_info,
        positional_cols=('surface_or_core', 'surface_or_core', 'anterior_surface', 'tail', 'donut'),
        conditions=('surface', 'core', True, True, True)
    ):
    df = df[df['nrtracks'] > 2]
    if 'treatment' in df.columns.values:
        df = df.drop('treatment', axis=1)
    df['treatment'] = df['path'].apply(get_treatment_name)
    df = scale_free_positional_categories(df, donut_df=donut_info)
    uframes = pd.unique(out['frame'])
    upaths = pd.unique(out['path'])
    its = len(upaths) * len(uframes)
    with tqdm(desc='Adding averages and count variables', total=its) as progress:
        for k, grp in df.groupby(['path', 'frame']):
            p = k[0]
            f = k[1]
            odf = out[(out['path'] == p) & (out['frame'] == f)]
            idx = odf.index.values
            out = _add_averages(out, grp, idx)
            ndf = df[(df['path'] == p) & (df['frame'] == f + 1)]
            for i, col in enumerate(positional_cols):
                cond = conditions[i]
                if isinstance(cond, str):
                    pre = cond + ' '
                else:
                    pre = col + ' '
                out = _add_averages(out, grp, idx, pre, col, cond)
                out = count_variables(out, grp, ndf, idx, col, cond, pre)
            progress.update(1)
    return out


def _add_averages(
        out, 
        df, 
        idx, 
        prefix='',
        column=None,
        condition=None,
        variables=('dv', 'dvz', 'dvy', 'dvx', 'ca_corr', 'dist_c', 'nb_density_15', 'ca_corr_pcnt', 'cont', 'elong'), 
        variable_names=('dv (um/s)', 'dvz (um/s)', 'dvy (um/s)', 'dvx (um/s)', 'corrected calcium', 
                        'centre distance', 'density (platelets/um^2)', 'corrected calcium (%)', 'contraction (um/s)', 'elongation')
    ):
    if column is not None:
        df = df[df[column] == condition]
    for i, v in enumerate(variables):
        n = prefix + variable_names[i]
        out.loc[idx, n] = df[v].mean()
    return out



# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def plot_averages(data, units):
    frame = data['frame']
    time = np.array(frame) / 0.321764322705706
    x = 'time (s)'
    y0 = f'radius {units}'
    hue = 'measure'
    # Data for plot 0
    # ---------------
    bd_time = np.concatenate([time, time.copy()])
    bd_radius = np.concatenate([data['births'], data['deaths']])
    birth = ['birth', ] * len(data['births'])
    death = ['death', ] * len(data['deaths'])
    bd_event = np.array(birth + death)
    bd_data = {
        x : bd_time, 
        y0 : bd_radius, 
        hue : bd_event
    }
    bd_data = pd.DataFrame(bd_data)
    # Data for plot 1
    # ---------------
    l_data = {
        x : time, 
        y0 : data['lifespan'], 
        hue : ['lifespan', ] * len(time)
    }
    l_data = pd.DataFrame(l_data)
    # Data for plot 2
    # ---------------
    y1 = 'Standard deviations from mean'
    o_data = {
        x : time, 
        y1 : data['outlierness'], 
        hue : ['outlierness', ] * len(time)
    }
    o_data = pd.DataFrame(o_data)
    _make_plot(x, y0, y1, hue, bd_data, l_data, o_data)



def _make_plot(x, y0, y1, hue, bd_data, l_data, o_data):
    sns.set_style("ticks")
    fig, axes = plt.subplots(3, 1, sharex=True)
    ax0, ax1, ax2 = axes.ravel()
    e0 = sns.lineplot(x, y0, data=bd_data, hue=hue, ax=ax0)
    e1 = sns.lineplot(x, y0, data=l_data, hue=hue, ax=ax1)
    e2 = sns.lineplot(x, y1, data=o_data, hue=hue, ax=ax2)
    #sns.despine()
    plt.show()



# -----------------------------------------------
# Multiple group comparison max barcode over time
# -----------------------------------------------


def largest_loop_comparison(
        paths, 
        save_path, 
        centile=75, 
        col='nd15_percentile', 
        y_col='ys_pcnt', 
        x_col='x_s_pcnt', 
        units='%'
    ):
    data = largest_loop_comparison_data(paths, save_path, centile, col, 
                                        y_col, x_col, units)
    tx_names = pd.unique(data['treatment'])
    plot_donut_comparison(data, treatments=tx_names)
    return data



def largest_loop_comparison_data(
        paths, 
        save_path, 
        centile=75, 
        col='nb_density_15_pcntf', 
        y_col='ys_pcnt', 
        x_col='x_s_pcnt', 
        units='%'
    ):
    out_df = []
    n = len(paths)
    started = False
    if os.path.exists(save_path):
        out_df = pd.read_csv(save_path)
        started = True
    for i, p in enumerate(paths):
        df = pd.read_parquet(p)
        print(f'Obtaining data for dataframe {i} of {n}')
        print(f'path: {p}')
        if 'nb_density_15' not in df.columns:
            df = add_neighbour_lists(df)
            df = local_density(df)      
        # do we need to add the quantile normalisation?
        if col not in df.columns:
            base = col[:-6]
            print('Quantile normalising ', base)
            df = quantile_normalise_variables_frame(df, [base, ])
        if 'x_s_pcnt' not in df.columns:
            print('Quantile normalising x coordinate')
            df = quantile_normalise_variables(df, ['x_s', ])
        if 'ys_pcnt' not in df.columns:
            print('Quantile normalising y coordinate')
            df = quantile_normalise_variables(df, ['ys', ])
        # is the information already saved from a previous attempt?
        do_df = True
        if os.path.exists(save_path):
            paths = pd.unique(df['path'])
            spaths = pd.unique(out_df['path'])
            bools = [p in spaths for p in paths]
            if False not in bools:
                do_df = False # don't repeat
        if do_df:
            out = largest_loop_data(df, centile=centile, col=col, y_col=y_col, 
                                x_col=x_col, get_accessory_data=True, 
                                units=units)
            key = Path(p).stem
            if started:
                out_df = pd.concat([out_df, out])
            else:
                out_df = out
                started = True
        out_df = out_df.reset_index(drop=True)
        out_df.to_csv(save_path)
    out_df = out_df.reset_index(drop=True)
    #spo = os.path.join(save_dir, save_name + '_outlierness.csv')
    out_df.to_csv(save_path)
    return out_df
    



def plot_donut_comparison(
        df, 
        treatments, 
        tx_col='treatment',
        time='time (s)', 
        persistence='persistence (%)', 
        donutness='difference from mean (std dev)'
    ):
    sml_df = [df[df[tx_col] == tx] for tx in treatments]
    sml_df = pd.concat(sml_df)
    sml_df = sml_df.reset_index(drop=True)
    fig, axs = plt.subplots(2, 1, sharex=True)
    ax0, ax1 = axs.ravel()
    sns.lineplot(time, persistence, hue=tx_col, hue_order=treatments, data=sml_df, ax=ax0)
    sns.lineplot(time, donutness, hue=tx_col, hue_order=treatments, data=sml_df, ax=ax1)
   #fig.set_size_inches()
    plt.show()


# --------------------
# Additional functions
# --------------------


def get_count(df, thresh_col='nd15_percentile', threshold=25):
    sml_df = df[df[thresh_col] > threshold]
    count = len(sml_df)
    return count


def get_treatment_name(inh): # need to rename from last run 
    if 'saline' in inh:
        out = 'saline'
    elif 'cang' in inh:
        out = 'cangrelor'
    elif 'veh-mips' in inh:
        out = 'MIPS vehicle'
    elif 'mips' in inh or 'MIPS' in inh:
        out = 'MIPS'
    elif 'sq' in inh:
        out = 'SQ'
    elif 'par4--biva' in inh:
        out = 'PAR4-- bivalirudin'
    elif 'par4--' in inh:
        out = 'PAR4--'
    elif 'biva' in inh:
        out = 'bivalirudin'
    elif 'SalgavDMSO' in inh or 'gavsalDMSO' in inh or 'galsavDMSO' in inh:
        out = 'DMSO (salgav)'
    elif 'Salgav' in inh or 'gavsal' in inh:
        out = 'salgav'
    elif 'DMSO' in inh:
        out = 'DMSO (MIPS)'
    elif 'dmso' in inh:
        out = 'DMSO (SQ)'
    elif 'ctrl' in inh:
        out = 'control'
    else:
        out = inh
    return out


# -----------------------------
#  Measurements of largest loop
# -----------------------------


def find_max_donut_time(
        df, 
        sn=None, 
        sd=None, 
        p='path', 
        tx='treatment', 
        t='time (s)', 
        y='difference from mean (std dev)', 
        lose=0.5, 
        thresh=6
    ):
    #adf = find_smoothed_average(df, tx, t, y)
    df = rolling_variable(df, p=p, t=t, y=y)
    y = y + ' rolling'
    summary = {
        tx :[],
        f'max {y} mean' : [], 
        f'max {y} SEM' : [], 
        f'{t} mean': [], 
        f'{t} SEM': [],
        'frames mean' : [], 
        'frames SEM' : [],
        f'time to minus {lose} mean': [], 
        f'time to minus {lose} SEM': [], 
        f'time over {thresh} mean' : [], 
        f'time over {thresh} SEM' : [], 
        'frame_min' : [], 
        'frame_max' : [] 
    }
    result = {
        p : [],
        tx :[], 
        f'max {y}' : [],
        t : [], 
        'frames' : [],
        f'time to minus {lose}' : [], 
        f'time over {thresh}' : [], 
    }
    for k, g in df.groupby([p, tx, ]):
        g = g.sort_values(t)
        peaks, props = find_peaks(g[y].values)
        idx = np.min(peaks)
        mv = g[y].values[idx]
        #mv = g[y].max()
        #idx = np.argmax(g[y].values)
        time = g[t].values[idx]
        result[p].append(k[0])
        result[tx].append(k[1])
        result[t].append(time)
        result[f'max {y}'].append(mv)
        f = np.round(time * 0.321764322705706).astype(int)
        result['frames'].append(f)
        # get time to lose 2
        smlg = g[g[t] > time]
        val = mv - lose
        i = np.where(smlg[y].values < val)
        if len(i[0]) > 0:
            min_i = np.min(i)
            ttm = smlg[t].values[min_i] - time
        else:
            ttm = np.inf
        result[f'time to minus {lose}'].append(ttm)
        # get time over 5
        ts = pd.unique(g[t])
        interval = ts[1] - ts[0]
        i = np.where(g[y] > thresh)
        to = len(i[0]) * interval
        result[f'time over {thresh}'].append(to)
    result = pd.DataFrame(result)
    for k, g in result.groupby([tx, ]):
        summary[tx].append(k)
        summary[f'max {y} mean'].append(g[f'max {y}'].mean())
        summary[f'max {y} SEM'].append(g[f'max {y}'].sem())
        summary[f'{t} mean'].append(g[t].mean())
        summary[f'{t} SEM'].append(g[t].sem())
        summary['frames mean'].append(g['frames'].mean())
        summary['frame_max'].append(np.round(g['frames'].mean() + 5).astype(int))
        summary['frame_min'].append(np.round(g['frames'].mean() - 3).astype(int))
        summary['frames SEM'].append(g['frames'].sem())
        summary[f'time to minus {lose} mean'].append(g[f'time to minus {lose}'].mean())
        summary[f'time to minus {lose} SEM'].append(g[f'time to minus {lose}'].sem())
        summary[f'time over {thresh} mean'].append(g[f'time over {thresh}'].mean())
        summary[f'time over {thresh} SEM'].append(g[f'time over {thresh}'].sem())
    summary = pd.DataFrame(summary)
    if sn is not None and sd is not None:
        sp0 = os.path.join(sd, sn +'_result.csv')
        result.to_csv(sp0)
        sp1 = os.path.join(sd, sn +'_summary.csv')
        summary.to_csv(sp1)
    return result, summary


def rolling_variable(df, p='path', t='time (s)', y='difference from mean (std dev)', time_average=False):
    n = y + ' rolling'
    for k, g in df.groupby([p]):
        g = g.sort_values(t)
        idx = g.index.values
        rolling = g[y].rolling(window=20,win_type='bartlett',min_periods=3,center=True).mean()
        df.loc[idx, n] = rolling
    return df



# --------------
# Simulated data
# --------------


def simulate_disc(sigma, size, r=100):
    sigma = r * sigma
    x = np.random.normal(scale=sigma, size=size)
    y = np.random.normal(scale=sigma, size=size)
    #y_pcnt = np.array([percentileofscore(y, v) for v in y])
    #x_pcnt = np.array([percentileofscore(x, v) for v in x])
    result = np.stack([x, y]).T
    return result



def simulate_1_loop(sigma, size, r=25):
    result = _loop_coords(size, sigma, r)
    return result



def simulate_2_loop(sigma, size):
    res_0 = _loop_coords(size, sigma, r=12.5)
    res_1 = _loop_coords(size, sigma, r=12.5)
    res_0 = res_0 + [-12.5, 0]
    res_1 = res_1 + [12.5, 0]
    result = np.concatenate([res_0, res_1])
    return result



def _loop_coords(n, s, r):
    s = r * s
    theta = np.linspace( 0 , 2 * np.pi , n)
    radius = r
    x = radius * np.cos( theta )
    y = radius * np.sin( theta )
    noise_x = np.random.normal(scale=s, size=n)
    noise_y = np.random.normal(scale=s, size=n)
    x = x + noise_x
    y = y + noise_y
    result = np.stack([x, y]).T
    result = result + 50
    return result


def generate_simulated_data(
        save_path,
        n=30, 
        sizes=(25, 50, 100, 200, 400, 800),
        sigmas=(0.2, 0.3, 0.4)
        ):
    df = {
        'sample_ID' : [],
        'distribution' : [],
        'sigma' : [],
        'size' : [],
        'sample_ID' : [],
        'x' : [], 
        'y' : [], 
    }
    func_dict = {
        'disc' : simulate_disc, 
        '1-loop' : simulate_1_loop,
        '2-loop' : simulate_2_loop
    }
    dists = list(func_dict.keys())
    sample_ID = 0
    its = len(sizes) * len(sigmas) * len(dists) * n
    with tqdm(total=its) as progress:
        for sigma in sigmas:
            for size in sizes:
                for dist in dists:
                    for i in range(n):
                        func = func_dict[dist]
                        result = func(sigma, size)
                        d = [dist, ] * len(result)
                        sig = [sigma, ] * len(result)
                        sz = [size, ] * len(result)
                        df['distribution'] = np.concatenate([df['distribution'], d])
                        df['sigma'] = np.concatenate([df['sigma'], sig])
                        df['size'] = np.concatenate([df['size'], sz])
                        df['x'] = np.concatenate([df['x'], result[:, 0]])
                        df['y'] = np.concatenate([df['y'], result[:, 1]])
                        sid = [sample_ID, ] * len(result)
                        df['sample_ID'] = np.concatenate([df['sample_ID'], sid])
                        sample_ID += 1
                        progress.update(1)
    df = pd.DataFrame(df)
    df.to_csv(save_path)


def simulated_data_matrix_plot(
        func='1_loop'
        sizes=(25, 50, 100, 200, 400, 800),
        sigmas=(0.2, 0.3, 0.4)    
    ):
    pass


# -------
# Helpers
# -------


def scale_x_and_y(df, x_col='x_s', y_col='ys'):
    df = df[df['nrtracks'] > 10]
    #if 'nb_density_15_pcntf' not in df.columns.values:
     #   df = quantile_normalise_variables_frame(df, ('nb_density_15', ))
    df = scale_data(df, x_col)
    df = scale_data(df, y_col)
    return df



def scale_data(df, col, groupby_list=['path', 'frame']):
    for k, g in df.groupby(groupby_list):
        scaler = StandardScaler()
        data = g[col].values
        data = np.expand_dims(data, 1)
        scaler.fit(data)
        new = scaler.transform(data)
        new = np.squeeze(new)
        idx = g.index.values
        df.loc[idx, f'{col}_scaled'] = new
    return df
