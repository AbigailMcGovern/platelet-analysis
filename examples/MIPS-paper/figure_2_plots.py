import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from toolz import curry
from plateletanalysis.variables.basic import quantile_normalise_variables
from plateletanalysis.variables.neighbours import local_density, add_neighbour_lists
from plateletanalysis.variables.transform import spherical_coordinates
import os
from pathlib import Path


def count(grp):
    return len(grp)


@curry
def density(dens_col, grp, *args):
    return np.mean(grp[dens_col].values)

@curry
def outer_edge(dist_col, pcnt_lims, grp, *args):
    grp = exclude_back_quadrant(grp)
    grp = quantile_normalise_variables(grp, [dist_col, ]) # does framewise within loop_over_exp
    dist_pcnt_col = dist_col + '_pcnt'
    grp = grp[(grp[dist_pcnt_col] > pcnt_lims[0]) & (grp[dist_pcnt_col] < pcnt_lims[1])]
    return np.mean(grp[dist_col].values)


def loop_over_exp(df, exp_col, time_col, val_col, val_func, other_cols):
    if isinstance(other_cols, str):
        other_cols = [other_cols, ]
    out = {
        exp_col : [],   
        val_col : [], 
        time_col : []           
    }
    for oc in other_cols:
        out[oc] = []
    for k, g in df.groupby([exp_col, time_col]): # would have done groupby apply, but I don't care
        val = val_func(g)
        out[exp_col].append(k[0])
        out[val_col].append(val)
        out[time_col].append(k[1])
        for oc in other_cols:
            out[oc].append(g[oc].values[0]) 
            # can add functionality here (mean) with optional extra arg
    out = pd.DataFrame(out)
    return out



def lineplots(
        df, 
        save_dir,
        save_names,
        exp_col='path',
        time_col='time (s)',
        track_lim=1,
        names=('platelet count', 'platelet density', 'thrombus edge distance'), 
        funcs=(count, density, outer_edge), 
        other_cols=('treatment', ), 
        curry_with=(None, 
                    ['nb_density_15', ], 
                    ['rho', (80, 98)]) 
        ):
    '''
    other_cols: tuple
        other columns to collect values for. Takes only the first value in group. 
        The first value in other_cols will be used as the hue for sns.lineplot. 
    '''
    df = df[df['nrtracks'] > track_lim]
    data_list = []
    
    for i, func in enumerate(funcs):
        n = names[i]
        cw = curry_with[i]
        if cw is not None:
            func = func(*cw)
        result = loop_over_exp(df, exp_col,time_col, n, func, other_cols)
        sp = os.path.join(save_dir, save_names[i])
        result.to_csv(sp) # we also need to make box plots from this data so save
        data_list.append(result)
        #TODO: separate script for plotting data
    # plots
    fig, axs = plt.subplots(1, len(names), sharex=True, sharey=False)
    for i, ax in enumerate(axs.ravel()):
        sns.lineplot(data=data_list[i], x=time_col, y=names[i], hue=other_cols[0], ax=ax, ci=70) #errorbar=("se", 2)) #, errorbar='se')
    plt.show()




def exclude_back_quadrant(df, col='phi', lim=- np.pi / 2):
    df = df[df[col] > lim]
    return df


def add_time_seconds(df, frame_col='frame'):
    df['time (s)'] = df[frame_col] / 0.321764322705706
    return df





def lineplots_data_all(
        df, 
        save_path,
        treatements=('MIPS', 'SQ', 'cangrelor'),
        controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline'),
        exp_col='path',
        time_col='time (s)',
        track_lim=10,
        names=('platelet count', 'platelet density', 'thrombus edge distance'), 
        funcs=(count, density, outer_edge), 
        other_cols=('treatment', ), 
        curry_with=(None, 
                    ['nb_density_15', ], 
                    ['rho', (90, 98)]) 
        ):
    '''
    other_cols: tuple
        other columns to collect values for. Takes only the first value in group. 
        The first value in other_cols will be used as the hue for sns.lineplot. 
    '''
    df = df[df['nrtracks'] > track_lim]
    data = []
    ind_save = [os.path.join(Path(save_path).parents[0], Path(save_path).stem + f'_{n}.csv') for n in names]
    for i, func in enumerate(funcs):
        if not os.path.exists(ind_save[i]):
            n = names[i]
            cw = curry_with[i]
            if cw is not None:
                func = func(*cw)
            result = loop_over_exp(df, exp_col,time_col, n, func, other_cols)
            result.to_csv(ind_save[i])
        else:
            result = pd.read_csv(ind_save[i])
        result = result.set_index([exp_col, time_col], drop=True)
        data.append(result)
        #TODO: separate script for plotting data
    data = pd.concat(data, axis=1)
    tx = data[other_cols[0]].values
    data = data.drop(columns=[other_cols[0], 'Unnamed: 0'])
    data[other_cols[0]] = tx[:, 0]
    data = data.reset_index(drop=False).drop_duplicates()
    data[:10].to_csv('/Users/abigailmcgovern/Data/platelet-analysis/MIPS/Figure_2/debugging.csv')
    # percentage data
    for n in names:
        nn = f'{n} pcnt'
        data[nn] = [None, ] * len(data)
    for i, tx in enumerate(treatements):
        v = controls[i]
        tx_df = data[data[other_cols[0]] == tx]
        for t, g in tx_df.groupby([time_col, ]):
            idxs = g.index.values
            v_df = data[(data[other_cols[0]] == v) & (data[time_col] == t)]
            v_means = {n : v_df[n].mean() for n in names}
            for n in names:
                nn = f'{n} pcnt'
                pcnt = g[n].values / v_means[n] * 100
                data.loc[idxs, nn] = pcnt
    data.to_csv(save_path)



def add_spherical_and_local_dens(data_dir, file_names):
    file_paths = [os.path.join(data_dir, n) for n in file_names]
    dfs = [pd.read_parquet(p) for p in file_paths]
    for p, df in zip(file_paths, dfs):
        if 'pid' not in df.columns.values:
            df['pid'] = range(len(df))
        if 'rho' not in df.columns.values:
           print(f'Getting spherical coords for {p}')
           df = spherical_coordinates(df)
           df.to_csv(p)
        if 'nb_density_15' not in df.columns.values:
           print(f'Getting neighbours for {p}')
           df = add_neighbour_lists(df)
           print(f'Getting density for {p}')
           df = local_density(df)
           df.to_csv(p)






if __name__ == '__main__':
    from plateletanalysis.variables.basic import get_treatment_name
    d = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes'
    mn = '211206_mips_df.parquet'
    vmn = '211206_veh-mips_df.parquet'
    mp = os.path.join(d, mn)
    vmp = os.path.join(d, vmn)
    # add local density and spherical - MIPS
    #m = pd.read_parquet(mp)
    #m = add_neighbour_lists(m)
    #m = local_density(m)
    #m = spherical_coordinates(m)
    #m.to_parquet(mp)
    # add local density and spherical - DMSO
    #vm = pd.read_parquet(vmp)
    #vm = add_neighbour_lists(vm)
    #vm = local_density(vm)
    #vm = spherical_coordinates(vm)
    #vm.to_parquet(vmp)
    # generate plots

    file_names = ('211206_mips_df.parquet', '211206_veh-mips_df.parquet', 
                  '211206_cang_df.parquet', '211206_saline_df_220827_amp0.parquet', 
                  '211206_sq_df.parquet', '211206_veh-sq_df.parquet')
    file_paths = [os.path.join(d, n) for n in file_names]
    #dfs = [pd.read_parquet(p) for p in file_paths]
    dfs = []
    for p in file_paths:
        df = pd.read_parquet(p)
        dfs.append(df)
    data = []
    for p, df in zip(file_paths, dfs):
        if 'pid' not in df.columns.values:
            print(f'Getting pid for {p}')
            df_len = len(df)
            df_pid = np.arange(df_len)
            df['pid'] = df_pid
        if 'rho' not in df.columns.values:
           print(f'Getting spherical coords for {p}')
           df = spherical_coordinates(df)
           df.to_parquet(p)
        if 'nb_density_15' not in df.columns.values:
           print(f'Getting neighbours for {p}')
           df = add_neighbour_lists(df)
           print(f'Getting density for {p}')
           df = local_density(df)
           df.to_parquet(p)
        data.append(df)
    df = pd.concat(data)
    df['treatment'] = df['path'].apply(get_treatment_name)
    df = df[df['treatment'] != 'DMSO (salgav)']
    df = add_time_seconds(df)
    save_dir = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/Figure_2'
    save_path = os.path.join(save_dir, 'counts_density_outeredge_MIPS_cang_biva_tl10_oe9098.csv')
    lineplots_data_all(df, save_path)

    #lineplots(df, 
     #         '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/Figure_2',
      #      ('MIPSvsDMSO_counts_1.csv', 'MIPSvsDMSO_density_1.csv', 'MIPSvsDMSO_outeredge_1.csv'))
    
    ## need to do the line plots with standard error bars rather than 95%CI (standard format)

