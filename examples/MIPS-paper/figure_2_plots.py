import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from toolz import curry
from plateletanalysis.variables.basic import quantile_normalise_variables
from plateletanalysis.variables.neighbours import local_density, add_neighbour_lists
from plateletanalysis.variables.transform import spherical_coordinates
import os


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
    # plots
    fig, axs = plt.subplots(1, len(names), sharex=True, sharey=False)
    for i, ax in enumerate(axs.ravel()):
        sns.lineplot(data=data_list[i], x=time_col, y=names[i], hue=other_cols[0], ax=ax) #, errorbar='se')
    plt.show()




def exclude_back_quadrant(df, col='phi', lim=- np.pi / 2):
    df = df[df[col] > lim]
    return df


def add_time_seconds(df, frame_col='frame'):
    df['time (s)'] = df[frame_col] / 0.321764322705706
    return df



if __name__ == '__main__':
    from plateletanalysis.variables.basic import get_treatment_name
    d = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes'
    mn = '211206_mips_df.parquet'
    vmn = '211206_veh-mips_df.parquet'
    mp = os.path.join(d, mn)
    vmp = os.path.join(d, vmn)
    # add local density and spherical - MIPS
    m = pd.read_parquet(mp)
    #m = add_neighbour_lists(m)
    #m = local_density(m)
    #m = spherical_coordinates(m)
    #m.to_parquet(mp)
    # add local density and spherical - DMSO
    vm = pd.read_parquet(vmp)
    #vm = add_neighbour_lists(vm)
    #vm = local_density(vm)
    #vm = spherical_coordinates(vm)
    #vm.to_parquet(vmp)
    # generate plots
    df = pd.concat([m, vm])
    df['treatment'] = df['path'].apply(get_treatment_name)
    df = df[df['treatment'] != 'DMSO (salgav)']
    df = add_time_seconds(df)
    lineplots(df, 
              '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/Figure_2',
            ('MIPSvsDMSO_counts.csv', 'MIPSvsDMSO_density.csv', 'MIPSvsDMSO_outeredge.csv'))
    
    ## need to do the line plots with standard error bars rather than 95%CI (standard format)

