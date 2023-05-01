import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from plateletanalysis.variables.basic import get_treatment_name, fsec_var, add_region_category, time_minutes, \
    add_nrtracks, add_phase, time_seconds, add_sliding_variable, tracking_time_var, time_tracked_var, add_terminating, \
    add_normalised_ca_pcnt, add_shedding, add_tracknr
import os
from tqdm import tqdm
from toolz import curry
import matplotlib

# --------
# Boxplots
# --------

def add_vars(df):
    df['treatment'] = df['path'].apply(get_treatment_name)
    df = df[df['treatment'] != 'DMSO (salgav)']
    print('Adding variables...')
    with tqdm(total=7) as progress: # ~ 1 min for first 7, add_shedding() takes ~ 5 hours
        df = time_seconds(df)
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


@curry
def bin_plts_long_short(tk_lim, trk_time):
    if trk_time > tk_lim:
        o = f'> {tk_lim} s'
    else:
        o = f'<= {tk_lim} s'
    return o


@curry
def bin_by_phase(phases, t):
    for phase in phases:
        s = f'{phase[0]}-{phase[1]} s'
        if t >= phase[0] and t < phase[1]:
            return s



def bin_by_c_al(b):
    if b == 'anterior' or b == 'lateral':
        return 'anteriolateral'
    elif b == 'center':
        return b
    else:
        return 'posterior'
    


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
    sns.set_context('paper')
    sns.set_style('ticks')
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
    matplotlib.rcParams.update({'font.size': 12})
    plt.show()

# ----------------
# First few tracks
# ----------------

def make_density_first_tracks_plots(df):
    #df['treatment'] = df['path'].apply(get_treatment_name)
    df = fsec_var(df)
    df = pd.concat([df[df['treatment'] == 'MIPS'], df[df['treatment'] == 'DMSO (MIPS)']])
    g = sns.catplot(data = df[df.tracknr <6], y='nb_density_15',x = 'tracknr', hue = 'treatment', col = 'fsec', row = 'region', 
            height = 2, aspect = 1, kind = 'point', errorbar = "se")#facet_kws={'sharey': False}, 
    #g.set(xscale="log")
    g.set(ylim= (6,16))
    g.set_titles("{col_name} \n {row_name} ")
    g.despine(top=True, right=True, left=True, bottom=True, offset=None, trim=False)
    plt.show()


def wrapper_function(dir, ns):
    #ns = [tx_n, ctr_n]
    df = [pd.read_parquet(os.path.join(dir, n)) for n in ns]
    df = pd.concat(df)
    df['treatment'] = df['path'].apply(get_treatment_name)
    df = add_region_category(df)
    make_density_first_tracks_plots(df)


if __name__ == '__main__':
    d = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes'
    first_part = True
    if first_part:
        ns = ['230301_MIPS_and_DMSO.parquet', '211206_mips_df.parquet', '211206_veh-mips_df.parquet']
        wrapper_function(d, ns)

    next_part = False
    if next_part:
        file_names = ('211206_mips_df.parquet', '211206_veh-mips_df.parquet', 
                      '211206_cang_df.parquet', '211206_saline_df_220827_amp0.parquet', 
                      '211206_sq_df.parquet', '211206_veh-sq_df.parquet', '230301_MIPS_and_DMSO.parquet')
        file_paths = [os.path.join(d, n) for n in file_names]
        data = []
        for p in file_paths:
            df = pd.read_parquet(p)
            if 'nrtracks' not in df.columns.values:
                df = add_nrtracks(df)
                df.to_parquet(p)
            if 'tracknr' not in df.columns.values:
                df = add_tracknr(df)
                df.to_parquet(p)
            df = add_vars(df)
            data.append(df)
        data = pd.concat(data).reset_index(drop=True)
        sp = os.path.join(d, 'MIPS_paper_data_all.parquet')
        data.to_parquet(sp)
        save_path = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/230319_initial_tracks/230320_first-10s_sp15_boxplot_200sp_regions1_data_8var_15sbin.csv'
        data = add_region_category(data)
        #print(data.columns.values)
        #data['region 1'] = data['region'].apply(bin_by_c_al)
        variables = ('ca_corr', 'Ca2+ pcnt max', 'sliding (ums^-1)', 'nb_density_15', 'stab', 'elong', 'cont', 'total time tracked (s)')
        out = data_for_ls_boxplots(data, variables, save_path, reg_var='region 1', trk_lim=15)
        boxplot_for_phases_regions(out, 'count', '> 15 s', sel_var='first 10 s', x_var='phase')
        boxplot_for_phases_regions(out, 'count', '<= 15 s', sel_var='first 10 s', x_var='phase')
        boxplot_for_phases_regions(out, 'nb_density_15', '> 15 s', sel_var='first 10 s', x_var='phase')
        boxplot_for_phases_regions(out, 'nb_density_15', '<= 15 s', sel_var='first 10 s', x_var='phase')
        boxplot_for_phases_regions(out, 'elong', '> 15 s', sel_var='first 10 s', x_var='phase')
        boxplot_for_phases_regions(out, 'elong', '<= 15 s', sel_var='first 10 s', x_var='phase')