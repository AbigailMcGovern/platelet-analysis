from tkinter.tix import Tree
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from plateletanalysis.variables.measure import quantile_normalise_variables
import numpy as np



def thrombus_overview_measures_comp(
        df, 
        save_path,
        thresh_col='nb_density_15_pcntf', 
        threshold=25, 
        centile_band=(75, 100),
        d_col='nb_density_15',
        d_pcnt_col='nb_density_15_pcntf',  
        r_col = 'dist_c',
        r_pcnt_col='rho_pcntf'
        ):
    out_df = {
        'treatment' : [], 
        'frame' : [],
        'path' : [],
        'time (s)' : [], 
        'platelet count' : [], 
        'turnover (%)' : [], 
        'local density (platelets/um^2)' : [], 
        'centre distance (um)' : []
    }
    df['time (s)'] = df['frame'] / 0.321764322705706
    out_df = get_numbers_turnover_density_rho(df, out_df, thresh_col='nd15_percentile', threshold=25, treatment_col='treatment')
    out_df.to_csv(save_path)
    make_comparative_plots(out_df)



def get_numbers_turnover_density_rho(
        df, 
        out_df, 
        thresh_col='nd15_percentile', 
        threshold=25, 
        treatment_col='treatment', 
        centile_band=(75, 100),
        d_col='nb_density_15',
        d_pcnt_col='nb_density_15_pcntf',  
        r_col = 'dist_c',
        r_pcnt_col='rho_pcntf'
        ):
    ups = pd.unique(df['path'])
    ufs = pd.unique(df['frame'])
    its = len(ups) * len(ufs)
    df_gb = df.groupby(['path', 'frame'])
    with tqdm(desc='Obtaining aggregate data', total=its) as progress:
        for key, grp in df_gb:
            p, f = key
            tx = get_treatment_name(p)
            sml_grp = grp[grp[thresh_col] > threshold]
            # ------------------
            # number and density
            # ------------------
            if len(sml_grp) > 0:
                n0 = len(sml_grp)
                frame = key[1]
                path = key[0]
                nxt_tp = (path, frame)
                sml_grp = []
                for k, g in df_gb:
                    if k == nxt_tp:
                        sml_grp = g[g[thresh_col] > threshold]
                if len(sml_grp) > 0:
                    n1 = len(sml_grp)
                    t = sml_grp['time (s)'].values[0]
                    turn = (n1 - n0) / n0 * 100
                    tx = sml_grp[treatment_col].values[0]
                else:
                    turn = np.NaN
                    t = np.NaN
                    tx = np.NaN
            # ------------------
            # centre and density
            # ------------------
            sml_grp = grp[(grp[d_pcnt_col] > centile_band[0]) & (grp[d_pcnt_col] < centile_band[1])]
            if len(sml_grp) > 0:
                d = sml_grp[d_col].mean()
            else:
                d = np.NaN
            sml_grp = grp[(grp[r_pcnt_col] > centile_band[0]) & (grp[r_pcnt_col] < centile_band[1])]
            if len(sml_grp) > 0:
                cd = sml_grp[r_col].mean()
            else:
                cd = np.NaN
            out_df['treatment'].append(tx)
            out_df['frame'].append(f)
            out_df['path'].append(p)
            out_df['time (s)'].append(t)
            out_df['platelet count'].append(n0)
            out_df['turnover (%)'].append(turn)
            out_df['local density (platelets/um^2)'].append(d)
            out_df['centre distance (um)'].append(cd)
            progress.update(1)
    out_df = pd.DataFrame(out_df)
    return out_df



def make_comparative_plots(out_df):
    fig, axes = plt.subplots(4, 1, sharex=True)
    ax0, ax1, ax2, ax3 = axes.ravel()
    sns.lineplot(x='time (s)', y='number', hue='treatment', palette='rocket', ax=ax0, data=out_df)
    sns.move_legend(ax0, "upper left", bbox_to_anchor=(1, 1))
    sns.lineplot(x='time (s)', y='turnover (%)', hue='treatment', palette='rocket', ax=ax1, data=out_df)
    sns.move_legend(ax1, "upper left", bbox_to_anchor=(1, 1))
    sns.lineplot(x='time (s)', y='local density (platelets/um^2)', hue='treatment', palette='rocket', ax=ax2, data=out_df)
    sns.move_legend(ax2, "upper left", bbox_to_anchor=(1, 1))
    sns.lineplot(x='time (s)', y='centre distance (um)', hue='treatment', palette='rocket', ax=ax3, data=out_df)
    sns.move_legend(ax3, "upper left", bbox_to_anchor=(1, 1))
    plt.show()



def get_treatment_name(inh):
    if 'saline' in inh:
        out = 'saline'
    elif 'biva' in inh:
        out = 'bivalirudin'
    elif 'cang' in inh:
        out = 'cangrelor'
    elif 'veh-mips' in inh:
        out = 'MIPS vehicle'
    elif 'mips' in inh:
        out = 'MIPS'
    elif 'sq' in inh:
        out = 'SQ'
    else:
        out = inh
    return out



if __name__ == '__main__':
    d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
    names = ['211206_saline_df_220827_amp0.parquet', '211206_biva_df.parquet', '211206_cang_df.parquet', '211206_sq_df.parquet', '211206_mips_df_220818.parquet', '211206_veh-mips_df_220831.parquet', '211206_veh-mips_df_220831.parquet']
    paths = [os.path.join(d, n) for n in names]
    from plateletanalysis.variables.measure import quantile_normalise_variables_frame
    cols_to_keep = ['frame', 'path', 'nb_density_15', 'nb_density_15_pcntf', 'dist_c', 'dist_c_pcntf']
    dfs = []
    for p in paths:
        df = pd.read_parquet(p)
        #df = quantile_normalise_variables_frame(df, ('dist_c', ))
        #if 'nb_density_15_pcntf' not in df.columns.values:
         #   df = quantile_normalise_variables_frame(df, ('nb_density_15', ))
        #df.to_parquet(p)
        df = df[cols_to_keep]
        dfs.append(df)
    df = pd.concat(dfs)
    df = df.reset_index(drop=True)

    df['treatment'] = df['path'].apply(get_treatment_name)

    thrombus_overview_measures_comp(df, n_thresh_col='nb_density_15_pcntf', n_threshold=25, 
                                    centile_band=(75, 100), d_pcnt_col='nb_density_15_pcntf',  
                                    r_pcnt_col='dist_c_pcntf')

