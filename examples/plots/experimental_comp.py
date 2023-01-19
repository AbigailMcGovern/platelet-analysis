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
    out_df = get_numbers_turnover_density_rho(df, out_df, thresh_col=thresh_col, threshold=threshold, treatment_col='treatment')
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
        r_pcnt_col='dist_c_pcntf'
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
                next_df = df[(df['path'] == p) & (df['frame'] == f + 1)]
                t = sml_grp['time (s)'].values[0]
                if len(next_df) > 0:
                    next_df = next_df[next_df[thresh_col] > threshold]
                    n1 = len(next_df)
                    turn = (n1 - n0) / n0 * 100
                else:
                    turn = np.NaN
            else:
                t = np.NaN
                n0 = np.NaN
                turn = np.NaN
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



def make_comparative_plots(out_df, hue_order=('saline', 'bivilarudin', 'cangrelor', 'SQ', 'MIPS', 'MIPS vehicle')):
    fig, axes = plt.subplots(4, 1, sharex=True)
    ax0, ax1, ax2, ax3 = axes.ravel()
    sns.lineplot(x='time (s)', y='platelet count', hue='treatment', hue_order=hue_order, ax=ax0, data=out_df)
    sns.move_legend(ax0, "upper left", bbox_to_anchor=(1, 1))
    line = ax1.axline((0, 0), (1, 0), ls='--', c='black', lw=1)
    sns.lineplot(x='time (s)', y='turnover (platetets)', hue='treatment', hue_order=hue_order, ax=ax1, data=out_df)
    sns.move_legend(ax1, "upper left", bbox_to_anchor=(1, 1))
    sns.lineplot(x='time (s)', y='local density (platelets/um^2)', hue='treatment', hue_order=hue_order, ax=ax2, data=out_df)
    sns.move_legend(ax2, "upper left", bbox_to_anchor=(1, 1))
    sns.lineplot(x='time (s)', y='centre distance (um)', hue='treatment', hue_order=hue_order, ax=ax3, data=out_df)
    sns.move_legend(ax3, "upper left", bbox_to_anchor=(1, 1))
    fig.subplots_adjust(right=0.77, left=0.125, bottom=0.05, top=0.99, hspace=0.1)
    fig.set_size_inches(6, 10)
    plt.show()
    return fig



def get_treatment_name(inh):
    if 'saline' in inh:
        out = 'saline'
    elif 'biva' in inh:
        out = 'bivalirudin'
    elif 'cang' in inh:
        out = 'cangrelor'
    elif 'veh-mips' in inh or 'DMSO' in inh:
        out = 'MIPS vehicle'
    elif 'mips' in inh or 'MIPS' in inh:
        out = 'MIPS'
    elif 'sq' in inh:
        out = 'SQ'
    else:
        out = inh
    return out


def combine_dataframes(paths, cols_to_keep = ['frame', 'path', 'nb_density_15', 'nb_density_15_pcntf', 'dist_c', 'dist_c_pcntf']):
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
    return df



def correct_turnover(df):
    df['count t1'] = (df['turnover (%)'] * df['platelet count'] / 100) + df['platelet count']
    paths = pd.unique(df['path'])
    for p in paths:
        pdf = df[df['path'] == p]
        idxs = pdf.index.values
        max_count = df['platelet count'].max()
        turnover = (pdf['count t1'] - pdf['platelet count']) / max_count * 100
        df.loc[idxs, 'turnover (% max)'] = turnover
    df['turnover (platetets)'] = df['count t1'] - df['platelet count']
    return df



if __name__ == '__main__':
    d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
    names = ['211206_saline_df_220827_amp0.parquet', '211206_biva_df.parquet', '211206_cang_df.parquet', '211206_sq_df.parquet', '211206_mips_df_220818.parquet', '211206_veh-mips_df_220831.parquet', '211206_veh-mips_df_220831.parquet']
    paths = [os.path.join(d, n) for n in names]
    #from plateletanalysis.variables.measure import quantile_normalise_variables_frame
    #cols_to_keep = ['frame', 'path', 'nb_density_15', 'nb_density_15_pcntf', 'dist_c', 'dist_c_pcntf']
    #dfs = []
    #for p in paths:
        #df = pd.read_parquet(p)
        #df = quantile_normalise_variables_frame(df, ('dist_c', ))
        #if 'nb_density_15_pcntf' not in df.columns.values:
         #   df = quantile_normalise_variables_frame(df, ('nb_density_15', ))
        #df.to_parquet(p)
        #df = df[cols_to_keep]
        #dfs.append(df)
    #df = pd.concat(dfs)
   # df = df.reset_index(drop=True)
    # debug with small data frame
    #df = df[df['frame'] < 10]

    #df = combine_dataframes(paths)
    #df['treatment'] = df['path'].apply(get_treatment_name)
    save_path = '/Users/amcg0011/Data/platelet-analysis/treatment_variable_comparison/saline_biva_cang_sq_mips_var-comp.csv'
    #thrombus_overview_measures_comp(df, save_path, thresh_col='nb_density_15_pcntf', threshold=25, 
     #                               centile_band=(75, 100), d_pcnt_col='nb_density_15_pcntf',  
      #                              r_pcnt_col='dist_c_pcntf')
    df = pd.read_csv(save_path)
    #print(df.head())
    #print(len(df))
    #print(df.columns.values)

    df = correct_turnover(df)
    df = df[df['frame'] < 191]

    # ------------------
    # Saline, Cang, Biva
    # ------------------

    def saline_cang_biva(df, sp):
        df = df[df['treatment'] != 'MIPS']
        df = df[df['treatment'] != 'MIPS vehicle']
        df = df[df['treatment'] != 'SQ']
        print(pd.unique(df['treatment']))
        sml_df = df[df['treatment'] == 'bivalirudin']
        print(len(sml_df)) # 3427
        print(sml_df['frame'].max()) # 190
        fig = make_comparative_plots(df, hue_order=('saline', 'bivalirudin', 'cangrelor'))
        fig.savefig(sp)
    
    sp = '/Users/amcg0011/Data/platelet-analysis/saline_biva_cang_var-comp-abs-turn.svg'
    
    #saline_cang_biva(df, sp)
    
    # --------------
    # DMSO, MIPS, SQ
    # --------------

    def dmso_MIPS_SQ(df, sp):
        df = df[df['treatment'] != 'saline']
        df = df[df['treatment'] != 'cangrelor']
        df = df[df['treatment'] != 'bivalirudin']
        print(pd.unique(df['treatment']))
        fig = make_comparative_plots(df)
        fig.savefig(sp)
    

    sp = '/Users/amcg0011/Data/platelet-analysis/dmso_MIPS_SQ_var-comp-abs-turn.svg'
    dmso_MIPS_SQ(df, sp)
