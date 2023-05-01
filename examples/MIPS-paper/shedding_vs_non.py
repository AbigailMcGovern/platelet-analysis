import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from toolz import curry
from scipy import stats

# ----------
# data funcs
# ----------

def first_2_frames(df):
    sdf = df[df['tracknr'] < 3]
    return sdf


def first_5_frames(df):
    sdf = df[df['tracknr'] < 6]
    return sdf



# -------------
# Main function
# -------------

def boxplots_data(
        df, 
        save_path, 
        variables, 
        funcs={'first 2 frames' : first_2_frames, 'first 5 frames' : first_5_frames}, 
        track_lim=1, 
        phases={'growth' : (0, 260), 'consolidation' : (260, 600)},
        groupby=['phase', 'path', 'shedding'], # can add region
        ):
    df = df[df['nrtracks'] > track_lim]
    df['phase'] = [None, ] * len(df)
    df = add_phase(df, phases)
    out = {c : [] for c in groupby}
    var_funcs = {}
    for v in variables:
        for f in funcs.keys():
            n = f"{v} : {f}"
            var_funcs[n] = (v, funcs[f])
            out[n] = []
    nits = 0
    for k, g in df.groupby(groupby):
        nits += 0
    with tqdm(total=nits) as progress:
        for k, g in df.groupby(groupby):
            for n in var_funcs.keys():
                for i, col in enumerate(groupby):
                    out[col].append(k[i])
                var, func = var_funcs[n]
                sdf = func(g)
                for v in variables:
                    out[var].append(sdf[v].mean())
            progress.update(1)
    out = pd.DataFrame(out)
    out.to_csv(save_path)
    return out



# -------------
# Add variables
# -------------


def add_nrtracks(df):
    for k, g in df.groupby(['path', 'particle', ]):
        n = len(g)
        idxs = g.index.values
        df.loc[idxs, 'nrtracks'] = n
    return df

def add_phase(df, phases={'growth' : (0, 260), 'consolidation' : (260, 600)}):
    df['phase'] = [None, ] * len(df)
    for phase in phases:
        sdf = df[(df['time (s)'] > phases[phase][0]) & (df['time (s)'] > phases[phase][1])]
        idxs = sdf.index.values
        df.loc[idxs, 'phase'] = phase
    return df



def add_time_seconds(df, frame_col='frame'):
    df['time (s)'] = df[frame_col] / 0.321764322705706
    return df


def add_sliding_variable(df):
    df['sliding (ums^-1)'] = [None, ] * len(df)
    # not moving in direction of blood flow
    sdf = df[df['dvy'] >= 0]
    idxs = sdf.index.values
    df.loc[idxs, 'sliding (ums^-1)'] = 0
    # moving in the direction of blood floe
    sdf = df[df['dvy'] < 0]
    idxs = sdf.index.values
    new = np.abs(sdf['dvy'].values)
    df.loc[idxs, 'sliding (ums^-1)'] = new
    return df


def tracking_time_var(df):
    df['tracking time (s)'] = df['tracknr'] / 0.321764322705706
    return df


def time_tracked_var(df):
    df['total time tracked (s)'] = df['nrtracks'] / 0.321764322705706
    return df


def add_terminating(df):
    df['terminating'] = [False, ] * len(df)
    for k, g in df.groupby(['path', ]):
        t_max = g['frame'].max()
        sdf = g[g['frame'] != t_max]
        term = sdf['nrtracks'] == sdf['tracknr']
        idxs = sdf.index.values
        df.loc[idxs, 'terminating'] = term
    return df


def add_normalised_ca_pcnt(df):
    for k, g in df.groupby(['path', 'frame']):
        ca_max = g['ca_corr'].max()
        ca_norm = g['ca_corr'] / ca_max * 100
        idxs = g.index.values
        df.loc[idxs, 'Ca2+ pcnt max'] = ca_norm
    return df


def add_shedding(df):
    df['shedding'] = [False, ] * len(df)
    nits = 0
    for k, g in df.groupby(['path', 'particle']):
        nits += 1
    with tqdm(total=nits) as progress:
        for k, g in df.groupby(['path', 'particle']):
            #shed = df['terminating'].sum() # was going to take 11 hours
            if True in df['terminating'].values:
                idxs = g.index.values
                df.loc[idxs, 'shedding'] = True
            progress.update(1)
    return df


def add_vars(df):
    df['treatment'] = df['path'].apply(get_treatment_name)
    df = df[df['treatment'] != 'DMSO (salgav)']
    print('Adding variables...')
    with tqdm(total=7) as progress: # ~ 1 min for first 7, add_shedding() takes ~ 5 hours
        df = add_time_seconds(df)
        progress.update(1)
        df = add_terminating(df)
        progress.update(1)
        df = add_sliding_variable(df)
        progress.update(1)
        df = time_minutes(df)
        progress.update(1)
        df = time_tracked_var(df)
        progress.update(1)
        df = tracking_time_var(df)
        progress.update(1)
        df = add_normalised_ca_pcnt(df)
        progress.update(1)
        #df = add_shedding(df)
        #progress.update(1)
    df.to_parquet(os.path.join(d, 'MIPS_paper_with_shedding.parquet'))
    return df


# --------
# Plotting
# --------


def plot_boxplot(
        data, 
        variables,
        treatment_col='treatment', 
        hue='shedding'
        ):
    fig, axs = plt.subplots(len(variables), 1)
    for i, ax in enumerate(axs.ravel()):
        sns.boxplot(data=data, x=treatment_col, y=variables[i], hue=hue, ax=ax)
    plt.show()


def violin_plots(
        data, 
        variable, 
        treatment_col='treatment', 
        hue='shedding'
        ):
    sns.violinplot(data=data, y=variable, x=treatment_col, hue=hue)
    plt.show()


def shedding_lineplot(data, variable, time_col='time (s)', hue='treatment'):
    sns.lineplot(data=data, x=time_col, y=variable, hue=hue)
    plt.show()


def shedding_lineplot(data, variable, time_col='time (s)', hue='treatment', treatments=('MIPS', 'SQ', 'cangrelor'), controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline')):
    fig, axs = plt.subplots(1, len(treatments), sharey=True)
    for i, ax in enumerate(axs.ravel()):
        tdf = data[data[hue] == treatments[i]]
        cdf = data[data[hue] == controls[i]]
        df = pd.concat([tdf, cdf]).reset_index(drop=True)
        sns.lineplot(data=df, y=variable, x=time_col, ax=ax, hue=hue)
    #sns.lineplot(data=data, x=time_col, y=variable, hue=hue)
    plt.show()


def start_boxplot_data(data, save_path, variables, phases=((0, 100), (100, 300), (300, 600)), tklim=30, start_len=10):
    data = data[data['nrtracks'] > 1]
    start_df_long = data[(data['total time tracked (s)'] > tklim) & (data['tracking time (s)'] <= start_len)]
    start_df_short = data[(data['total time tracked (s)'] <= tklim) & (data['tracking time (s)'] <= start_len)]
    out = {
        'path' : [],
        'treatment' : [], 
        'phase' : [], 
        f'first {start_len} s tracked' : [] 
    }
    for v in variables:
        out[v] = []
    len_strs = [f'> {tklim} s', f'<= {tklim} s']
    phase_strs = [f'{p[0]}-{p[1]} s' for p in phases]
    for i, phase in enumerate(phases):
        start_df_long_p = start_df_long[(start_df_long['time (s)'] >= phase[0]) & (start_df_long['time (s)'] < phase[1])]
        start_df_short_p = start_df_short[(start_df_short['time (s)'] >= phase[0]) & (start_df_short['time (s)'] < phase[1])]
        for p in pd.unique(start_df_long['path']):
            long = start_df_long_p[start_df_long_p['path'] == p]
            short = start_df_short_p[start_df_short_p['path'] == p]
            ls_dfs = [long, short]
            for j in range(2):
                out['path'].append(p)
                out['treatment'].append(long['treatment'].values[0])
                out['phase'].append(phase_strs[i])
                out[f'first {start_len} s tracked'].append(len_strs[j])
                for v in variables:
                    val = ls_dfs[j][v].mean()
                    out[v].append(val)
    out = pd.DataFrame(out)
    out.to_csv(save_path)
    return out


def start_boxplots(out, var, tx='treatment', hue='first 10 s tracked'):
    sns.boxplot(data=out, x=tx, y=var, hue=hue)
    plt.show()


def kde_for_first_few_sec_data(data, variables, save_path, phases=((0, 100), (100, 300), (300, 600)), start_len=10):
    data = data[data['nrtracks'] > 1]
    data = data[data['tracking time (s)'] <= start_len]
    out = {
        'treatment' : [], 
        'path' : [], 
        'particle' : [], 
        'phase' : []
    }
    for v in variables:
        out[v] = []
    for phase in phases:
        phase_str = f'{phase[0]}-{phase[1]} s'
        df = data[(data['time (s)'] >= phase[0]) & (data['time (s)'] < phase[1])]
        for k, g in df.groupby(['path', 'particle']):
            out['path'].append(k[0])
            out['particle'].append(k[1])
            out['phase'].append(phase_str)
            out['treatment'].append(g['treatment'].values[0])
            for v in variables:
                out[v].append(g[v].mean())
    out = pd.DataFrame(out)
    out.to_csv(save_path)
    return out


def kde_for_last_few_sec_data(data, variables, save_path, phases=((0, 100), (100, 300), (300, 600)), end_len=10):
    data = data[data['nrtracks'] > 1]
    data = data[data['frame'] < 191]
    frames = end_len * 0.321764322705706
    data['nfromend'] = data['nrtracks'] - data['tracknr']
    data = data[data['nfromend'] <= frames]
    out = {
        'treatment' : [], 
        'path' : [], 
        'particle' : [], 
        'phase' : []
    }
    for v in variables:
        out[v] = []
    for phase in phases:
        phase_str = f'{phase[0]}-{phase[1]} s'
        df = data[(data['time (s)'] >= phase[0]) & (data['time (s)'] < phase[1])]
        for k, g in df.groupby(['path', 'particle']):
            out['path'].append(k[0])
            out['particle'].append(k[1])
            out['phase'].append(phase_str)
            out['treatment'].append(g['treatment'].values[0])
            for v in variables:
                out[v].append(g[v].mean())
    out = pd.DataFrame(out)
    out.to_csv(save_path)
    return out



def kde_for_first_few_sec(
        out, 
        y_var, 
        treatments=('MIPS', 'SQ', 'cangrelor'), 
        controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline'), 
        x_var='total time tracked (s)', 
        hue='treatment'
        ):
    phases = pd.unique(out['phase'])
    fig, axs = plt.subplots(len(treatments), len(phases), sharex='col', sharey=True)
    for i in range(len(treatments)):
        tx = treatments[i]
        ctrl = controls[i]
        data = pd.concat([out[out['treatment'] == tx], out[out['treatment'] == ctrl]])
        for j in range(len(phases)):
            phase = phases[j]
            sml_data = data[data['phase'] == phase]
            ax = axs[i, j]
            sns.kdeplot(data=sml_data, x=x_var, y=y_var, ax=ax, hue=hue)
            ax.set_xscale('log')
    plt.show()


def last_few_frames_boxplots(out, var):
    data = {
        'treatment' : [], 
        'path' : [], 
        'phase' : [], 
        'last 10 s' : [],
        var : []
    }
    lsbins = out['total time tracked (s)'].apply(bin_as_long_short)
    out['last 10 s'] = lsbins
    phases = pd.unique(out['phase'])
    for k, grp in out.groupby(['path', 'treatment', 'phase', 'last 10 s']):
        data['path'].append(0)
        data['treatment'].append(k[1])
        data['phase'].append(k[2])
        data['last 10 s'].append(k[3])
        data[var].append(grp[var].mean())
    data = pd.DataFrame(data)
    fig, axs = plt.subplots(1, len(phases), sharey=True)
    for i, ax in enumerate(axs.ravel()):
        p = phases[i]
        sd = data[data['phase'] == p]
        sns.boxplot(x=var, y='treatment', hue='last 10 s', data=sd, ax=ax)
        ax.set_title(p)
    plt.show()


def bin_as_long_short(trk_time, tk_lim=30):
    if trk_time > tk_lim:
        o = f'> {tk_lim} s'
    else:
        o = f'<= {tk_lim} s'
    return o

@curry
def bin_plts_long_short(tk_lim, trk_time):
    if trk_time > tk_lim:
        o = f'> {tk_lim} s'
    else:
        o = f'<= {tk_lim} s'
    return o


def start_time_plots_data(data, variables, save_path, start_len=10):
    data = data[data['nrtracks'] > 1]
    data = data[data['tracking time (s)'] <= start_len]
    lsbins = data['total time tracked (s)'].apply(bin_as_long_short)
    data[f'first {start_len} s'] = lsbins
    bin_str = f'first {start_len} s'
    out = {
        'treatment' : [], 
        'path' : [], 
        'time (s)' : [], 
        bin_str : [],
    }
    for v in variables:
        out[v] = []
    gb_list = ['path', 'treatment', 'time (s)', bin_str]
    for k, grp in data.groupby(gb_list):
        for i, col in enumerate(gb_list):
            out[col].append(k[i])
        for v in variables:
            out[v].append(grp[v].mean())
    out = pd.DataFrame(out)
    out.to_csv(save_path)
    return out



def timeplots(out, var, treatments=('MIPS', 'SQ', 'cangrelor'), 
                controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline')):
    fig, axs = plt.subplots(1, len(treatments))
    for i, ax in enumerate(axs.ravel()):
        tx = out[out['treatment'] == treatments[i]]
        v = out[out['treatment'] == controls[i]]
        df = pd.concat([tx, v])
        sns.lineplot(data=df, x='time (s)', y=var, ax=ax, hue='treatment', style='first 10 s')
        ax.set_title(treatments[i])
    plt.show()


def add_region_category(df):
    rcyl = (df.x_s ** 2 + df.ys ** 2) ** 0.5
    df['rcyl'] = rcyl
    df['region'] = [None, ] * len(df)
    # center
    sdf = df[df['rcyl'] <= 37.5]
    idxs = sdf.index.values
    df.loc[idxs, 'region'] = 'center'
    # outer regions
    # 45 degrees = 0.785398
    sdf = df[df['rcyl'] > 37.5]
    # anterior
    rdf = sdf[sdf['phi'] > 0.785398]
    idxs = rdf.index.values
    df.loc[idxs, 'region'] = 'anterior'
    # lateral
    rdf = sdf[(sdf['phi'] < 0.785398) & (sdf['phi'] > -0.785398)]
    idxs = rdf.index.values
    df.loc[idxs, 'region'] = 'lateral'
    # posterior
    rdf = sdf[sdf['phi'] < -0.785398]
    idxs = rdf.index.values
    df.loc[idxs, 'region'] = 'posterior'
    return df


def start_time_plots_regions_data(data, variables, save_path, start_len=10, trk_lim=30):
    data = data[data['nrtracks'] > 1]
    data = data[data['tracking time (s)'] <= start_len]
    data = add_region_category(data)
    bin_func = bin_plts_long_short(trk_lim)
    lsbins = data['total time tracked (s)'].apply(bin_func)
    data[f'first {start_len} s'] = lsbins
    bin_str = f'first {start_len} s'
    out = {
        'treatment' : [], 
        'path' : [], 
        'minute' : [], 
        bin_str : [],
        'region' : [], 
        'count' : []
    }
    for v in variables:
        out[v] = []
    gb_list = ['path', 'treatment', 'minute', 'region', bin_str]
    for k, grp in data.groupby(gb_list):
        for i, col in enumerate(gb_list):
            out[col].append(k[i])
        for v in variables:
            out[v].append(grp[v].mean())
        out['count'].append(len(pd.unique(grp['particle'])))
    out = pd.DataFrame(out)
    out.to_csv(save_path)
    return out


def start_time_plots_regions(out, var, treatments=('MIPS', 'SQ', 'cangrelor'), 
                controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline'), regions=('center', 'anterior', 'lateral', 'posterior')):
    fig, axs = plt.subplots(len(treatments), len(regions), sharex=True, sharey=True)
    for i, tx in enumerate(treatments):
        tdf = out[out['treatment'] == tx]
        cdf = out[out['treatment'] == controls[i]]
        df = pd.concat([tdf, cdf])
        for j, region in enumerate(regions):
            sdf = df[df['region'] == region]
            sns.lineplot(data=sdf, x='minute', y=var, ax=axs[i, j], hue='treatment', style='first 10 s', marker='o', err_style='bars', ci=75)
            axs[i, j].set_title(f'{tx}: {region}')
            #axs[i, j].set_yscale('log')
    fig.subplots_adjust(right=0.97, left=0.07, bottom=0.075, top=0.93, wspace=0.3, hspace=0.3)
    plt.show()


@curry
def bin_by_phase(phases, t):
    for phase in phases:
        s = f'{phase[0]}-{phase[1]} s'
        if t >= phase[0] and t < phase[1]:
            return s
    

def data_for_ls_boxplots(
        data, 
        variables, 
        save_path, 
        phases=((0, 200), (200, 400), (400, 600)),  
        start_len=10, 
        trk_lim=15, 
        reg_var='region'):
    data = data[data['nrtracks'] > 1]
    data = data[data['tracking time (s)'] <= start_len]
    bin_func = bin_plts_long_short(trk_lim)
    lsbins = data['total time tracked (s)'].apply(bin_func)
    data[f'first {start_len} s'] = lsbins
    bin_func = bin_by_phase(phases)
    pbins = data['time (s)'].apply(bin_func)
    data['phase'] = pbins
    bin_str = f'first {start_len} s'
    out = {
        'treatment' : [], 
        'path' : [], 
        'phase' : [], 
        bin_str : [],
        reg_var : [], 
        'count' : [], 
        'pcnt of all platelets' : [], 
        'pcnt of region platelets' : []
    }
    for v in variables:
        out[v] = []
    gb_list = ['path', 'treatment', 'phase', reg_var, bin_str]
    reg_list = ['path', 'treatment', 'phase', reg_var]
    p_counts = {p : len(pd.unique(data[data['path'] == p]['particle'])) for p in pd.unique(data['path'])}
    p_r_counts = {k : len(pd.unique(grp['particle'])) for k, grp in data.groupby(reg_list)}
    for k, grp in data.groupby(gb_list):
        for i, col in enumerate(gb_list):
            out[col].append(k[i])
        for v in variables:
            out[v].append(grp[v].mean())
        n_plt = len(pd.unique(grp['particle']))
        n_region_plt = p_r_counts[(k[0], k[1], k[2], k[3])]
        out['count'].append(n_plt)
        pcnt_of_all = n_plt / p_counts[k[0]] * 100
        out['pcnt of all platelets'].append(pcnt_of_all)
        out['pcnt of region platelets'].append(n_plt / n_region_plt * 100)
    out = pd.DataFrame(out)
    out.to_csv(save_path)
    return out



def boxplot_for_phases_regions(
        out, 
        y_var, 
        selection,
        x_var='first 10 s', 
        hue_var='treatment', 
        sel_var='phase', 
        treatments=('MIPS', 'SQ', 'cangrelor'), 
        controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline'), 
        regions=('center', 'anterior', 'lateral', 'posterior'), 
        reg_var='region'
        ):
    fig, axs = plt.subplots(len(regions), len(treatments), sharey='row')
    for i, tx in enumerate(treatments):
        tdf = out[out['treatment'] == tx]
        cdf = out[out['treatment'] == controls[i]]
        df = pd.concat([tdf, cdf])
        df = df[df[sel_var] == selection]
        hue_order = [controls[i], tx]
        for j, region in enumerate(regions):
            sdf = df[df[reg_var] == region]
            sns.boxplot(data=sdf, x=x_var, y=y_var, ax=axs[j, i], hue=hue_var, hue_order=hue_order, width=0.6)
            axs[j, i].set_title(f'{tx}: {region}')
            #axs[i, j].set_yscale('log')
    fig.subplots_adjust(right=0.95, left=0.15, bottom=0.08, top=0.92, wspace=0.25, hspace=0.43)
    plt.show()



def violinplot_for_phases_regions(
        out, 
        y_var, 
        selection,
        x_var='first 10 s', 
        hue_var='treatment', 
        sel_var='phase', 
        treatments=('MIPS', 'SQ', 'cangrelor'), 
        controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline'), 
        regions=('center', 'anterior', 'lateral', 'posterior')
        ):
    fig, axs = plt.subplots(len(regions), len(treatments), sharey='row')
    for i, tx in enumerate(treatments):
        tdf = out[out['treatment'] == tx]
        cdf = out[out['treatment'] == controls[i]]
        df = pd.concat([tdf, cdf])
        df = df[df[sel_var] == selection]
        hue_order = [controls[i], tx]
        for j, region in enumerate(regions):
            sdf = df[df['region'] == region]
            sns.violinplot(data=sdf, x=x_var, y=y_var, ax=axs[j, i], hue=hue_var, hue_order=hue_order, width=0.6)
            axs[j, i].set_title(f'{tx}: {region}')
            #axs[i, j].set_yscale('log')
    fig.subplots_adjust(right=0.95, left=0.15, bottom=0.08, top=0.92, wspace=0.25, hspace=0.43)
    plt.show()



def add_pcnt_max_col(df, var, treatments=('MIPS', 'SQ', 'cangrelor'), controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline')):
    n = f'{var} pcnt'
    df[n] = [None, ] * len(df)
    for i, tx in enumerate(treatments):
        tdf = df[df['treatment'] == tx]
        vdf = df[df['treatment'] == controls[i]]
        v_mean = vdf[var].mean()
        pcnt_v = tdf[var] / v_mean * 100
        idxs = tdf.index.values
        df.loc[idxs, n] = pcnt_v
    return df



def boxplot_for_treatment_pcnt(
        out, 
        y_var, 
        x_var='phase', 
        hue_var='treatment', 
        sel_var='first 10 s', 
        treatments=('MIPS', 'SQ', 'cangrelor'), 
        regions=('center', 'anterior', 'lateral', 'posterior'), 
        reg_var='region'
    ):
    df = pd.concat([out[out['treatment'] == tx] for tx in treatments])
    selection = pd.unique(df[sel_var])
    fig, axs = plt.subplots(len(selection), len(regions), sharey='row')
    for i, sel in enumerate(selection):
        data = df[df[sel_var] == sel]
        for j, region in enumerate(regions):
            rdata = data[data[reg_var] == region]
            sns.boxplot(x=x_var, y=y_var, hue=hue_var, data=rdata, ax=axs[i, j], width=0.7)
            axs[i, j].set_title(f'{sel}: {region}')
    fig.subplots_adjust(right=0.95, left=0.125, bottom=0.07, top=0.95, wspace=0.2, hspace=0.45)
    plt.show()



def bin_by_c_al(b):
    if b == 'anterior' or b == 'lateral':
        return 'anteriolateral'
    elif b == 'center':
        return b
    else:
        return 'posterior'
    

def get_mannwhitneyu_data(
        out, 
        variables, 
        save_path, 
        sel_var='first 10 s',
        treatments=('MIPS', 'SQ', 'cangrelor'), 
        controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline'), 
        regions=('anteriolateral', 'center'), 
        reg_var='region 1', 
        alternative='less', 
        trk_lim=15, 
        start_len=10
        ):
    data = {
        'treatment' : [], 
        reg_var : [], 
        sel_var : [], 
        'phase' : [],
    }
    for v in variables:
        data[v] = []
        data[f'{v}: Mann-Whitney U'] = []
        data[f'{v}: p-value'] = []
    gb = [reg_var, sel_var, 'treatment', 'phase']
    tdf = pd.concat([out[out['treatment'] == tx] for tx in treatments])
    vdf = pd.concat([out[out['treatment'] == v] for v in controls])
    vdfs = {}
    for k, g in vdf.groupby(gb):
        nk = (k[0], k[1])
        vdfs[nk] = g
    for k, g in tdf.groupby(gb):
        nk = (k[0], k[1])
        veh = vdfs[nk]
        for col in gb:
            data[col].append(g[col].values[0])
        for v in variables:
            data[v].append(g[v].mean())
            U, P = stats.mannwhitneyu(g[v].values, veh[v].values, alternative=alternative)
            data[f'{v}: Mann-Whitney U'].append(U)
            data[f'{v}: p-value'].append(P)
    data = pd.DataFrame(data)
    data.to_csv(save_path)
    return data


def bin_by_lifespan(sec):
    bins = ((0, 15), (15, 30), (30, 60), (60, 120), (120, 600))
    for l, h in bins:
        if sec >= l and sec < h:
            return f'{l}-{h} s'


def lineplots_across_platelet_lifespan(
        data, 
        var, 
        treatments=('MIPS', 'SQ', 'cangrelor'), 
        controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline'), 
        regions=('center', 'anterior', 'lateral', 'posterior')
        ):
    bins = data['total time tracked (s)'].apply(bin_by_lifespan)
    data['lifespan'] = bins
    x = 'tracking time (s)'
    line_plot_treatment_by_region(data, x, var, treatments, controls, regions)
            

def line_plot_treatment_by_region(
        data,
        x, 
        y, 
        treatments=('MIPS', 'SQ', 'cangrelor'), 
        controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline'), 
        regions=('center', 'anterior', 'lateral', 'posterior')
        ):
    fig, axs = plt.subplots(len(regions), len(treatments), sharey=True)
    for i, tx in enumerate(treatments):
        tdf = data[data['treatment'] == tx]
        cdf = data[data['treatment'] == controls[i]]
        df = pd.concat([tdf, cdf])
        for j, region in enumerate(regions):
            rdf = df[df['region'] == region]
            sns.lineplot(x=x, y=y, data=rdf, hue='lifespan', style='treatment', ax=axs[j, i])
            axs[j, i].set_title(f'{tx}: {region}')
    plt.show()



def lifespan_vs_var_scatter_data(data, save_path, variables, start_len=10, end_len=10, phases=((0, 200), (200, 400), (400, 600))):
    data = data[data['nrtracks'] > 1]
    frames = end_len * 0.321764322705706
    data['nfromend'] = data['nrtracks'] - data['tracknr']
    data = data[data['nfromend'] <= frames]
    data = add_region_category(data)
    bin_func = bin_by_phase(phases)
    pbins = data['time (s)'].apply(bin_func)
    data['phase'] = pbins
    out = {
        'treatment' : [], 
        'path' : [], 
        'phase' : [],
        'total time tracked (s)' : [],
        'region' : [], 
        'count' : [], 
        'pcnt of all platelets' : [], 
        'pcnt of region platelets' : []
    }
    for v in variables:
        out[v] = []
        out[f'{v}: first {start_len}s'] = []
        out[f'{v}: last {end_len}s'] = []
    gb_list = ['path', 'treatment', 'region', 'phase', 'total time tracked (s)']
    reg_list = ['path', 'treatment', 'region', 'phase']
    p_counts = {p : len(pd.unique(data[data['path'] == p]['particle'])) for p in pd.unique(data['path'])}
    p_r_counts = {k : len(pd.unique(grp['particle'])) for k, grp in data.groupby(reg_list)}
    for k, grp in data.groupby(gb_list):
        # categorical
        for i, col in enumerate(gb_list):
            value = k[i]
            out[col].append(value)
        # mean values
        for v in variables:
            out[v].append(grp[v].mean())
        # start values
        start = grp[grp['tracking time (s)'] <= start_len]
        for v in variables:
            out[f'{v}: first {start_len}s'].append(start[v].mean())
        # end values
        end = grp[grp['nfromend'] <= frames]
        for v in variables:
            out[f'{v}: last {end_len}s'].append(end[v].mean())
        n_plt = len(pd.unique(grp['particle']))
        n_region_plt = p_r_counts[(k[0], k[1], k[2], k[3])]
        # count vars
        out['count'].append(n_plt)
        pcnt_of_all = n_plt / p_counts[k[0]] * 100
        out['pcnt of all platelets'].append(pcnt_of_all)
        out['pcnt of region platelets'].append(n_plt / n_region_plt * 100)
    out = pd.DataFrame(out)
    out.to_csv(save_path)
    return out


def scatter_treatment_by_region(
        data,
        x, 
        y, 
        treatments=('MIPS', 'SQ', 'cangrelor'), 
        controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline'), 
        regions=('center', 'anterior', 'lateral', 'posterior')
        ):
    fig, axs = plt.subplots(len(regions), len(treatments), sharey=True)
    for i, tx in enumerate(treatments):
        tdf = data[data['treatment'] == tx]
        cdf = data[data['treatment'] == controls[i]]
        df = pd.concat([tdf, cdf])
        for j, region in enumerate(regions):
            rdf = df[df['region'] == region]
            sns.lineplot(x=x, y=y, data=rdf, hue='treatment', ax=axs[j, i])
            axs[j, i].set_title(f'{tx}: {region}')
    plt.show()

if __name__ == '__main__':
    from plateletanalysis.variables.basic import get_treatment_name, time_minutes

    # --------
    # Get data
    # --------
    # takes about 1-2 min to read in data 
    d = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes'
    file_names = ('211206_mips_df.parquet', '211206_veh-mips_df.parquet', 
                  '211206_cang_df.parquet', '211206_saline_df_220827_amp0.parquet', 
                  '211206_sq_df.parquet', '211206_veh-sq_df.parquet', '230301_MIPS_and_DMSO.parquet')
    file_paths = [os.path.join(d, n) for n in file_names]
    #dfs = [pd.read_parquet(p) for p in file_paths]
    data = []
    for p in file_paths:
        df = pd.read_parquet(p)
        if 'nrtracks' not in df.columns.values:
            df = add_nrtracks(df)
        df = add_vars(df)
        data.append(df)
    data = pd.concat(data).reset_index(drop=True)
    #del data
    sp = os.path.join(d, 'MIPS_paper_data_all.parquet')
    data.to_parquet(sp)


    # -------------
    # Add variables
    # -------------
    # df = add_vars(df)
    #data = pd.read_parquet(os.path.join(d, 'MIPS_paper_with_shedding.parquet'))

    # ----------------------
    # Obtain data for graphs
    # ----------------------
    sns.set_context('paper')
    sns.set_style('ticks')
    save_path = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/shedding_vs_non/230212_sheddingvsnon_growth_consol_.csv'
    variables = ('ca_corr', 'Ca2+ pcnt max', 'sliding (ums^-1)', 'total time tracked (s)', 'nb_density_15', 'stab')
    variables = ('ca_corr', 'Ca2+ pcnt max', 'sliding (ums^-1)', 'nb_density_15', 'stab', 'elong', 'cont', 'total time tracked (s)')
    #out = boxplots_data(df, save_path, variables) 
    #out = pd.read_csv(save_path)
    #plot_boxplot(out, ('Ca2+ pcnt max', 'sliding (ums^-1)', 'nb_density_15'))
    #gdf = data[data['time (s)'] < 260]
    #cdf = data[data['time (s)'] >= 260]
    #violin_plots(gdf, 'total time tracked (s)', hue='minute')
    #violin_plots(cdf, 'total time tracked (s)', hue='minute')
    #shed_df = data[data['terminating'] == True]
    #start_df_long = data[(data['nrtracks'] > 10) & (data['tracknr'] == 1)]
    #start_df_short = data[(data['nrtracks'] < 10) & (data['tracknr'] == 1)]
    #shedding_lineplot(shed_df, 'total time tracked (s)')
    #shedding_lineplot(shed_df, 'Ca2+ pcnt max')
    #variables = ('ca_corr', 'Ca2+ pcnt max', 'sliding (ums^-1)', 'total time tracked (s)', 'nb_density_15')
    save_path = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/shedding_vs_non/230216_starttrack_comp.csv'
    #out = start_boxplot_data(data, save_path, variables)
    #out = pd.read_csv(save_path)
    #for phase, g in out.groupby('phase'):
    #    print(phase)
      #  for var in variables:
       #     start_boxplots(g, var)
    #save_path = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/shedding_vs_non/230221_first-10s_platelet_data_5var.csv'
    #out = kde_for_first_few_sec_data(data, variables, save_path)
    #out = pd.read_csv(save_path)
    #print(pd.unique(out.phase))
    #kde_for_first_few_sec(out, 'sliding (ums^-1)')

    #save_path = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/shedding_vs_non/230221_last-10s_platelet_data_5var.csv'
    #out = kde_for_last_few_sec_data(data, variables, save_path)
    #out = pd.read_csv(save_path)
    #kde_for_first_few_sec(out, 'Ca2+ pcnt max')
    #last_few_frames_boxplots(out, 'total time tracked (s)')
    #save_path = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/shedding_vs_non/230221_first-10s_timeplot_data_5var.csv'
    #out = start_time_plots_data(data, variables, save_path)
    #out = pd.read_csv(save_path)
    #timeplots(out, 'Ca2+ pcnt max')
    #save_path = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/shedding_vs_non/230221_first-10s_timeplot_min_regions_data_7var_10sbin.csv'
    #out = start_time_plots_regions_data(data, variables, save_path, trk_lim=15)
    #out = pd.read_csv(save_path)
    #start_time_plots_regions(out, 'count')
    #start_time_plots_regions(out, 'nb_density_15')
    #start_time_plots_regions(out, 'Ca2+ pcnt max')
    #start_time_plots_regions(out, 'stab')
    #start_time_plots_regions(out, 'total time tracked (s)')
    save_path = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/shedding_vs_non/230221_first-10s_sp15_boxplot_200sp_regions1_data_8var_15sbin.csv'
    #out = data_for_ls_boxplots(data, variables, save_path)
    data = add_region_category(data)
    #print(data.columns.values)
    data['region 1'] = data['region'].apply(bin_by_c_al)
    out = data_for_ls_boxplots(data, variables, save_path, reg_var='region 1', trk_lim=15)
    #out = pd.read_csv(save_path)
    #boxplot_for_phases_regions(out, 'count', '0-200 s')
    #boxplot_for_phases_regions(out, 'count', '200-400 s')
    #boxplot_for_phases_regions(out, 'count', '400-600 s')
    #boxplot_for_phases_regions(out, 'nb_density_15', '0-200 s')
    #boxplot_for_phases_regions(out, 'nb_density_15', '200-400 s')
    #boxplot_for_phases_regions(out, 'nb_density_15', '400-600 s')

    #out = add_pcnt_max_col(out, 'count')
    #out = add_pcnt_max_col(out, 'nb_density_15')
    #out = add_pcnt_max_col(out, 'pcnt of all platelets')
    #out['region 1'] = out['region'].apply(bin_by_c_al)
    #out = add_pcnt_max_col(out, 'pcnt of region platelets')

    boxplot_for_phases_regions(out, 'count', '> 15 s', sel_var='first 10 s', x_var='phase')
    boxplot_for_phases_regions(out, 'count', '<= 15 s', sel_var='first 10 s', x_var='phase')
    boxplot_for_phases_regions(out, 'nb_density_15', '> 15 s', sel_var='first 10 s', x_var='phase')
    boxplot_for_phases_regions(out, 'nb_density_15', '<= 15 s', sel_var='first 10 s', x_var='phase')
    boxplot_for_phases_regions(out, 'elong', '> 15 s', sel_var='first 10 s', x_var='phase')
    boxplot_for_phases_regions(out, 'elong', '<= 15 s', sel_var='first 10 s', x_var='phase')

    #boxplot_for_treatment_pcnt(out, 'count pcnt', regions=('center', 'anteriolateral'), reg_var='region 1')
    #boxplot_for_treatment_pcnt(out, 'pcnt of all platelets pcnt', regions=('center', 'anteriolateral'), reg_var='region 1')
    #boxplot_for_treatment_pcnt(out, 'nb_density_15 pcnt', regions=('center', 'anteriolateral'), reg_var='region 1')
    #boxplot_for_treatment_pcnt(out, 'pcnt of region platelets pcnt', regions=('center', 'anteriolateral', 'posterior'), reg_var='region 1')
    
    #boxplot_for_phases_regions(out, 'pcnt of region platelets', '<= 15 s', sel_var='first 10 s', x_var='phase',
    #                           regions=('center', 'anteriolateral'), reg_var='region 1')
    #boxplot_for_phases_regions(out, 'pcnt of region platelets', '> 15 s', sel_var='first 10 s', x_var='phase',
    #                           regions=('center', 'anteriolateral'), reg_var='region 1')
    
    vartup = ('nb_density_15', 'pcnt of all platelets', 'pcnt of region platelets')
    sp = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/shedding_vs_non/230226_MannWhitneyU_density-pcntall-pcntreg_2reg_15sbin_higher.csv'
    #sdf = get_mannwhitneyu_data(out, vartup, sp, alternative='greater')

    #lineplots_across_platelet_lifespan(data, 'ca_corr')
    variables = ('ca_corr', 'Ca2+ pcnt max', 'sliding (ums^-1)', 'nb_density_15', 'stab', 'elong', 'cont')
    save_path = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/shedding_vs_non/230301_scatter-data_total-time-vs-11vars_mean-start-end.csv'
    #out = lifespan_vs_var_scatter_data(data, save_path, variables)
    #out = pd.read_csv(save_path)
    #scatter_treatment_by_region(out, 'total time tracked (s)', 'nb_density_15')
    #scatter_treatment_by_region(out, 'total time tracked (s)', 'nb_density_15: first 10s')
    #scatter_treatment_by_region(out, 'total time tracked (s)', 'nb_density_15: last 10s')
    #scatter_treatment_by_region(out, 'total time tracked (s)', 'stab')
    #scatter_treatment_by_region(out, 'total time tracked (s)', 'stab: first 10s')
    #scatter_treatment_by_region(out, 'total time tracked (s)', 'stab: last 10s')
    #scatter_treatment_by_region(out, 'total time tracked (s)', 'sliding (ums^-1)')
    #scatter_treatment_by_region(out, 'total time tracked (s)', 'sliding (ums^-1): first 10s')
    #scatter_treatment_by_region(out, 'total time tracked (s)', 'sliding (ums^-1): last 10s')
    #scatter_treatment_by_region(out, 'total time tracked (s)', 'Ca2+ pcnt max')
    #scatter_treatment_by_region(out, 'total time tracked (s)', 'Ca2+ pcnt max: first 10s')
    #scatter_treatment_by_region(out, 'total time tracked (s)', 'Ca2+ pcnt max: last 10s')

    #save_path = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/shedding_vs_non/230221_first-10s_sp15_boxplot_200sp_regions1_data_8var_15sbin.csv'