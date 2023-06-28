import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt



MIPS_order = ['DMSO (MIPS)', 'MIPS']
cang_order = ['saline','cangrelor']#['Saline','Cangrelor','Bivalirudin']
SQ_order = ['DMSO (SQ)', 'SQ']
pal_MIPS  =dict(zip(MIPS_order, sns.color_palette('Blues')[2::3]))
pal_cang = dict(zip(cang_order, sns.color_palette('Oranges')[2::3]))
pal_SQ = dict(zip(SQ_order, sns.color_palette('Greens')[2::3]))
pal1={**pal_MIPS,**pal_cang,**pal_SQ}



def angle_plot(
        df: pd.DataFrame, 
        variables, 
        abs_treatement='MIPS', 
        abs_control='DMSO (MIPS)',  
        pcnt_treatments=('MIPS', 'SQ', 'cangrelor'), 
        pcnt_controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline'), 
        ):
    x = 'angle from midpoint (degrees)'
    df = df.sort_values(x).copy()
    #print(df[x].values[0], df[variables[0]].values[0])
    for v in variables:
        _add_rolled(v, df)
    df = pcnt_veh(variables, pcnt_treatments, pcnt_controls, df)
    pcnt_cols = [f'{var} (% vehicle)' for var in variables]
    # first row will be abs data for mips and dmso
    # second row will be percentage data
    fig, axs = plt.subplots(2, len(variables), subplot_kw={'projection' : 'polar'})
    for i, v in enumerate(variables):
        ax = axs[0, i]
        sdf = pd.concat([df[df['treatment'] == abs_control],
                          df[df['treatment'] == abs_treatement]])
        #print(sdf[x].values[0], sdf[v].values[0])
        sns.lineplot(data=sdf, x='angle from midpoint (degrees)', y=v, ax=ax, 
                     hue='treatment', hue_order=(abs_control, abs_treatement), 
                     errorbar=("se", 1), palette=pal1)
        _add_quadrant_lines(sdf, v, ax)
        ax.set_xlim(-90, 90)
        #print(v)
    for i, v in enumerate(pcnt_cols):
        ax = axs[1, i]
        dfs = [df[df['treatment'] == tx] for tx in pcnt_treatments]
        sdf = pd.concat(dfs)
        sns.lineplot(data=sdf, x='angle from midpoint (degrees)', y=v, ax=ax, 
                     hue='treatment', hue_order=pcnt_treatments, 
                     errorbar=("se", 1), palette=pal1)
        ax.set_xlim(-90, 90)
        _add_quadrant_lines(sdf, v, ax)
        #print(v)
    sns.despine()
    fig.subplots_adjust(right=0.95, left=0.1, bottom=0.1, top=0.95, wspace=0.4, hspace=0.2)
    fig.set_size_inches(14, 5.5)
    plt.show()


def pcnt_veh(vars, treatments, controls, df):
    cols = [f'{var} (% vehicle)' for var in vars]
    for tx, ctl in zip(treatments, controls):
        ctl_df = df[df['treatment'] == ctl].copy()
        sdf = pd.concat([df[df['treatment'] == tx], ctl_df]).copy()
        for k, grp in sdf.groupby('angle from midpoint (degrees)'): 
            idxs = grp.index.values
            for var, n in zip(vars, cols):
                if sdf[var].min() < 0:
                    min_val = sdf[var].min()
                    sdf[var] = sdf[var] - min_val
                    ctl_df[var] = ctl_df[var] - min_val
                var_veh_mean = ctl_df[ctl_df['angle from midpoint (degrees)'] == k][var].mean()
                orig = grp[var].values
                #vals = orig / var_veh_mean * 100
                df.loc[idxs, n] = orig / var_veh_mean * 100
    return df


def _add_quadrant_lines(sdf, v, ax):
    v_min = sdf[v].mean() - sdf[v].sem()
    v_max = sdf[v].mean() + sdf[v].sem()
    if v_max == v_min:
        v_min = sdf[v].min()
        v_max = sdf[v].max()
    ax.axline((-45, v_min), (-45, v_max), color='grey', alpha=0.5, ls='--')
    ax.axline((45, v_min), (45, v_max), color='grey', alpha=0.5, ls='--')
    


def _add_rolled(col, df):
    for k, grp in df.groupby('path'):
        idxs = grp.index.values
        roll = grp[col].rolling(window=5,center=True).mean()
        df.loc[idxs, col] = roll


variables = ('platelet count', 'platelet density gain (um^-3)', 
                   'platelet average density (um^-3)', 'frame average density (um^-3)',
                   'average platelet stability', 'recruitment', 
                   'P(recruited < 15 s)', 'total sliding (um)', 
                   'average platelet sliding (um)', 'average platelet contraction (um s^-1)', 
                   'average contraction in frame (um s^-1)', 'average platelet corrected calcium',
                   'average frame corrected calcium', 'shedding', 
                   'average platelet tracking time (s)', 'P(< 15s)')


vars = ('recruitment (s^-1)', 'P(recruited < 15 s)', 
        'platelet density gain (um^-3)', 'average contraction in frame (um s^-1)',
        'average platelet sliding (um)', 'average platelet stability')

vars_list = [('platelet count', 'platelet density gain (um^-3)', 
                   'platelet average density (um^-3)', 'frame average density (um^-3)',), 
           ('average platelet stability', 'recruitment', 
                   'P(recruited < 15 s)', 'total sliding (um)'),
            ('average platelet sliding (um)', 'average platelet contraction (um s^-1)', 
                   'average contraction in frame (um s^-1)', 'average platelet corrected calcium'),  
                   ('average frame corrected calcium', 'shedding', 
                   'average platelet tracking time (s)', 'P(< 15s)')]

save_path = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/Figure_3/230512_angle_25bin_outside_inj_data.csv'
#save_path = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/Figure_3/230512_angle_40bin_f3f_outside_inj_data.csv'
df = pd.read_csv(save_path)
df = df[df['phase'] == 'consolidation']
#for v in vars_list:
   # angle_plot(df, v)
df = df.rename(columns={'average platelet stability' : 'average platelet instability'})
for k, grp in df.groupby('treatment'):
    print(k)
    print(pd.unique(grp.path))
vars = ('platelet count', 'platelet average density (um^-3)', 'average platelet instability')
angle_plot(df, vars)

