import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import ptitprince as pt
import seaborn as sns

# Run this scrip using the raincloudplots environment 
# newer versions of numpy have no np.bool, which is needed for ptitprince


MIPS_order = ['DMSO (MIPS)', 'MIPS']
cang_order = ['saline','cangrelor']#['Saline','Cangrelor','Bivalirudin']
SQ_order = ['DMSO (SQ)', 'SQ']
pal_MIPS  =dict(zip(MIPS_order, sns.color_palette('Blues')[2::3]))
pal_cang = dict(zip(cang_order, sns.color_palette('Oranges')[2::3]))
pal_SQ = dict(zip(SQ_order, sns.color_palette('Greens')[2::3]))
pal1={**pal_MIPS,**pal_cang,**pal_SQ}


def pcnt_veh(gb, vars, treatments, controls, df):
    cols = [f'{var} (% vehicle)' for var in vars]
    for tx, ctl in zip(treatments, controls):
        ctl_df = df[df['treatment'] == ctl].copy()
        sdf = pd.concat([df[df['treatment'] == tx], ctl_df]).copy()
        for var in vars:
            min_val = sdf[var].min()
            if min_val < 0:
                snv = sdf[var] - min_val
                sdf[var] = snv
                cnv = ctl_df[var] - min_val
                ctl_df[var] = cnv
        for k, grp in sdf.groupby(gb): 
            idxs = grp.index.values
            for var, n in zip(vars, cols):
                orig = grp[var].values
                var_veh_mean = ctl_df[ctl_df[gb] == k][var].mean()
                vals = orig / var_veh_mean * 100
                df.loc[idxs, n] = vals
                if vals.min() < 0:
                    pass
    return df


# ------------------
# Get data from file
# ------------------
#psp = f'/Users/abigailmcgovern/Data/platelet-analysis/MIPS/Figure_3/230420_count-and-growth-pcnt_peaks.csv'
#peaks = pd.read_csv(psp)
#save_path = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/Figure_4/230601_exp_region_phase_plot_data.csv'
save_path = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/summary_data/230612_exp_region_phase_data.csv'
result = pd.read_csv(save_path)
result['average platelet sliding (um)'] = result['average platelet sliding (um)'] / 0.32
result = result[result['phase'] == 'consolidation']

#result['initial platelet decceleration (um/s)'] = - result['initial platelet velocity change (um/s)']
#result['total change in velocity (um/s)'] = - result['total change in velocity (um/s)']


#result = result.rename(columns={'average platelet instability' : 'average platelet instability', })

variables = ('platelet count', 'platelet density gain (um^-3)', 
             'platelet average density (um^-3)', 'frame average density (um^-3)',
             'average platelet instability', 'recruitment', 
             'P(recruited < 15 s)', 'total sliding (um)', 
             'average platelet sliding (um)', 'average platelet contraction (um s^-1)', 
             'average contraction in frame (um s^-1)', 'average platelet corrected calcium',
             'average frame corrected calcium', 'shedding', 
             'average platelet tracking time (s)', 'initial corrected calcium', 
             'initial platlet density (um^-3)', 'initial platelet instability', 
             'angle at time shed', 'y coordinate at time shed', 
             'distance from centre at time shed', 'intial platelet density gain (um^-3)', 
             'P(< 15s)', 'initial platelet velocity change (um/s)', 
             'average net platelet loss (/min)', 'average greatest net platelet loss (/min)')


vars_list = [
    ('platelet count', 'platelet density gain (um^-3)', 
             'platelet average density (um^-3)', 'frame average density (um^-3)',), 
    ('average platelet instability', 'recruitment', 
             'P(recruited < 15 s)', 'total sliding (um)', ), 
    ('average platelet sliding (um)', 'average platelet contraction (um s^-1)', 
             'average contraction in frame (um s^-1)', 'average platelet corrected calcium',), 
    ('average frame corrected calcium', 'shedding', 
             'average platelet tracking time (s)', 'initial corrected calcium', ), 
    ('initial platlet density (um^-3)', 'initial platelet instability', 
             'angle at time shed', 'y coordinate at time shed', ),
    ('distance from centre at time shed', 'intial platelet density gain (um^-3)', 
             'P(< 15s)', 'initial platelet velocity change (um/s)'), 
    ('average net platelet loss (/min)', 'average greatest net platelet loss (/min)', 
             'average platelet elongation', 'total change in velocity (um/s)'), 
    ('average platelet y velocity (um/s)', 'P(> 60s)', 
                   'P(recruited > 60 s)', 'platelets from other regions')
]



def raincloud_plots_with_comp(
        df, 
        vars, 
        abs_treatment='MIPS', 
        abs_control='DMSO (MIPS)',  
        pcnt_treatments=('MIPS', 'SQ', 'cangrelor'), 
        pcnt_controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline'), 
        ):
    ort = "h"; sigma = .4; move = .3
    fig, axs = plt.subplots(len(vars), 2, sharey=True)
    result  = pcnt_veh('region', vars, pcnt_treatments, pcnt_controls, df)
    pcnt_cols = [f'{var} (% vehicle)' for var in vars]
    for i, v in enumerate(vars):
        # ---------
        # Raincloud
        # ---------
        ax = axs[i, 0]
        data = pd.concat([result[result['treatment'] == abs_control], 
                          result[result['treatment'] == abs_treatment]])
        order = ('center', 'anterior', 'lateral', 'posterior')
        pt.RainCloud(x='region', y=v, hue='treatment', palette=pal1, order=order,
                     hue_order=('MIPS', 'DMSO (MIPS)'), data=data, bw = sigma,
                    width_viol = .7, orient = ort, move=move, alpha=0.8, ax=ax)
        ax.legend([],[], frameon=False)
        sns.despine()
        # ----------------
        # Comparative plot
        # ----------------
        ax = axs[i, 1]
        pv = pcnt_cols[i]
        data = pd.concat([result[result['treatment'] == tx] for tx in pcnt_treatments])
        sns.boxplot(y='region', x=pv, hue='treatment', palette=pal1, 
                    hue_order=('MIPS', 'SQ', 'cangrelor'), 
                    data=data, ax=ax, order=order)
        ax.legend([],[], frameon=False)
        sns.despine()
    fig.subplots_adjust(right=0.9, left=0.18, bottom=0.07, top=0.97, wspace=0.3, hspace=0.5)
    fig.set_size_inches(6.5, 9)
    plt.show()



def raincloud_plots(
        df, 
        vars, 
        save=None, 
        abs_treatment='MIPS', 
        abs_control='DMSO (MIPS)',  
        ):
    ort = "h"; sigma = .4; move = .3
    n_rows = int(np.ceil(len(vars)/4))
    fig, axs = plt.subplots(n_rows, 4, sharey=True)
    for i, v in enumerate(vars):
        # ---------
        # Raincloud
        # ---------
        if n_rows > 1:
            row = int(np.ceil(i / 4) - 1)
            column = i % 4
            ax = axs[row, column]
        else:
            ax = axs[i]
        data = pd.concat([df[df['treatment'] == abs_control], 
                          df[df['treatment'] == abs_treatment]])
        order = ('center', 'anterior', 'lateral', 'posterior')
        pt.RainCloud(x='region', y=v, hue='treatment', palette=pal1, order=order,
                     hue_order=('MIPS', 'DMSO (MIPS)'), data=data, bw = sigma,
                    width_viol = .7, orient = ort, move=move, alpha=0.8, ax=ax)
        ax.legend([],[], frameon=False)
        sns.despine()
    fig.subplots_adjust(right=0.98, left=0.071, bottom=0.175, top=0.98, wspace=0.165, hspace=0.5)
    fig.set_size_inches(14.5, n_rows * 3.5)
    if save is not None:
        fig.savefig(save)
    plt.show()


def quadrant_raincloud_plots(
        df, 
        vars, 
        abs_treatment='MIPS', 
        abs_control='DMSO (MIPS)', 
        quadrants=('anterior', 'lateral', 'posterior'), 
        log=False 
        ):
    ort = "h"; sigma = .4; move = .3
    n_vars = len(vars)
    fig, axs = plt.subplots(n_vars, 3, sharey=True)
    for i, v in enumerate(vars):
        # ---------
        # Raincloud
        # ---------
        for j, q in enumerate(quadrants):
            if n_vars > 1:
                row = i
                column = j
                ax = axs[row, column]
            else:
                ax = axs[j]
            sdf = df[df['quadrant'] == q]
            data = pd.concat([sdf[sdf['treatment'] == abs_control], 
                              sdf[sdf['treatment'] == abs_treatment]])
            order = ('0-25 um', '25-45 um', '45-65 um')
            pt.RainCloud(x='rho_bin', y=v, hue='treatment', palette=pal1, order=order,
                         hue_order=('MIPS', 'DMSO (MIPS)'), data=data, bw = sigma,
                        width_viol = .7, orient = ort, move=move, alpha=0.8, ax=ax)
            ax.legend([],[], frameon=False)
            if log:
                ax.set_xscale('log')
            sns.despine()
    fig.subplots_adjust(right=0.98, left=0.13, bottom=0.15, top=0.98, wspace=0.165, hspace=0.5)
    fig.set_size_inches(10, n_vars * 3)
    plt.show()


def stripplots(
        df, 
        vars, 
        abs_treatment='MIPS', 
        abs_control='DMSO (MIPS)',  
        ):
    fig, axs = plt.subplots(len(vars), 1, sharey=False)
    for i, v in enumerate(vars):
        ax = axs[i]
        order = ('center', 'anterior', 'lateral', 'posterior')
        data = pd.concat([df[df['treatment'] == abs_control], 
                          df[df['treatment'] == abs_treatment]])
        sns.stripplot(x='region', y=v, hue='treatment', palette=pal1, order=order,
                     hue_order=('MIPS', 'DMSO (MIPS)'), data=data, ax=ax, dodge=True)
        ax.legend([],[], frameon=False)
        sns.despine()
    fig.subplots_adjust(right=0.9, left=0.18, bottom=0.07, top=0.97, wspace=0.3, hspace=0.5)
    fig.set_size_inches(6.5, 9)
    plt.show()


#for v in vars_list:
    #raincloud_plots_with_comp(result, vars=v)


#variables = ('recruitment', 'P(recruited < 15 s)', 'average platelet instability', 'initial platelet velocity change (um/s)', )
#raincloud_plots(result, variables)
#variables = ('recruitment', 'P(recruited < 15 s)', 'platelet density gain (um^-3)', 'average platelet instability')
#raincloud_plots_with_comp(result, vars=variables)

vars_dict = {
    'platelet count' : 'count', 
    'platelet average density (um^-3)' : 'density', 
    'platelet density gain (um^-3)' : 'density gain', 
    'average platelet instability' : 'instability', 
    'average net platelet loss (/min)' : 'net platelet loss',
    'average platelet tracking time (s)' : 'time tracked',
    'P(< 15s)' : 'P(< 15s)',
    'P(> 60s)' : 'P(> 60s)', 
    'recruitment' : 'recruitment', 
    'shedding' : 'shedding', 
    'P(recruited < 15 s)' : 'P(recruited < 15 s)', 
    'P(recruited > 60 s)': 'P(recruited > 60 s)', 
    'average platelet y velocity (um/s)' : 'y-axis velocity', 
    'total change in velocity (um/s)' : 'net decceleration', 
    'initial platlet density (um^-3)' : 'initial density', 
    'initial platelet instability' : 'initial instability',
    'initial platelet velocity change (um/s)' : 'initial decelleration', 
}
variables = [k for k in vars_dict.keys()]
save = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/all_raincloud.pdf'
#raincloud_plots(result, variables, save)


save_path = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/summary_data/230623_exp_cylrad_binned_phase_data.csv'
df = pd.read_csv(save_path)
df['average platelet sliding (um)'] = df['average platelet sliding (um)'] / 0.32
df = df[df['phase'] == 'consolidation']
variables = ('average platelet y velocity (um/s)', )
quadrant_raincloud_plots(df, variables)



#injs = ['210520_IVMTR109_Inj2_DMSO_exp3', '210520_IVMTR109_Inj3_DMSO_exp3', '210520_IVMTR109_Inj4_DMSO_exp3', '210520_IVMTR109_Inj6_DMSO_exp3']
#result.loc[result.path.isin (injs), 'initial corrected calcium'] = np.nan
#result.loc[result.path.isin (injs), 'average platelet corrected calcium'] = np.nan
#result.loc[result.path.isin (injs), 'average frame corrected calcium'] = np.nan


#variables = ('platelet count', 'platelet density gain (um^-3)' , 'initial platelet velocity change (um/s)', 'initial corrected calcium')
#raincloud_plots_with_comp(result, vars=variables)

# calcium
#variables = ( 'initial corrected calcium', 'average platelet corrected calcium', 'average frame corrected calcium')
#raincloud_plots_with_comp(result, vars=variables)

#variables = ('recruitment', 'P(recruited < 15 s)', 'average net platelet loss (/min)',)
#for k, g in result.groupby('treatment'):
#    print(k)
#    print(pd.unique(g.path))
#raincloud_plots_with_comp(result, vars=variables)
#stripplots(result, vars=variables)