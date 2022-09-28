
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from plateletanalysis.variables.measure import quantile_normalise_variables, quantile_normalise_variables_frame
import numpy as np
from tqdm import tqdm
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols


def timeplot_surface_comparison(df0, df1, y_col='dv', threshold=60, thresh_col='nb_density_15_pcntf',  zs=7, ys=0):
    df0['time (s)'] = df0['frame'] / 0.321764322705706
    df0['treatment'] = ['MIPS', ] * len(df0)
    df1['time (s)'] = df1['frame'] / 0.321764322705706
    df1['treatment'] = ['vehicle ', ] * len(df1)
    df = pd.concat([df0, df1])
    del df0
    del df1
    df = df.reset_index(drop=True)
    df = add_surface_variables(df, threshold=threshold, thresh_col=thresh_col, zs=zs, ys=ys)
    fig, axes = plt.subplots(2, 1, sharex=True)
    ax0, ax1 = axes.ravel()
    sns.lineplot(data=df, x='time (s)', y=y_col, hue='treatment', palette='rocket', lw=1, ax=ax0, style='surface')
    sns.move_legend(ax0, "upper left", bbox_to_anchor=(1, 1))
    sns.lineplot(data=df, x='time (s)', y=y_col, hue='treatment', palette='rocket', lw=1, ax=ax1, style='anterior_surface')
    sns.move_legend(ax1, "upper left", bbox_to_anchor=(1, 1))
    plt.show()



def add_surface_variables(df, threshold=60, thresh_col='nb_density_15_pcntf', zs=7, ys=0):
    ant_df = df[df['ys'] > 0]
    pos_df = df[df['ys'] < 0]
    ant_df = quantile_normalise_variables_frame(ant_df, ('dist_c', )) # quantile normailise for both anterior and posterior
    pos_df = quantile_normalise_variables_frame(pos_df, ('dist_c', )) # asymmetric around y axis, symetric around x so no need to worry about this
    df = pd.concat([ant_df, pos_df])
    df = df.reset_index(drop=True)
    quantile_normalise_variables_frame
    df['surface'] = ['core', ] * len(df)
    sdf = df[(df[thresh_col] < threshold) & (df['zs'] > 15) & (df['dist_c_pcntf'] > 40)]
    s_idxs = sdf.index.values
    df.loc[s_idxs, 'surface'] = ['surface', ] * len(sdf)
    df['anterior_surface'] = ['core and posterior', ] * len(df)
    asdf = sdf[sdf['ys'] > ys]
    as_idxs = asdf.index.values
    df.loc[as_idxs, 'anterior_surface'] = ['anterior surface', ] * len(asdf)
    print(pd.unique(df['surface']))
    print(pd.unique(df['anterior_surface']))
    return df


def generate_mips_data(
        m_df,
        v_df, 
        save_dir,
        save_name,
        time_bins={
            'growth phase' : (0, 180), 
            'transition phase' : (180, 260), 
            'consolidation phase': (260, 600)
        }, 
        threshold=60, 
        thresh_col='nb_density_15_pcntf',  
        zs=15, 
        ys=0
    ):
    m_df['time (s)'] = m_df['frame'] / 0.321764322705706
    m_df['treatment'] = ['MIPS', ] * len(m_df)
    v_df['time (s)'] = v_df['frame'] / 0.321764322705706
    v_df['treatment'] = ['vehicle ', ] * len(v_df)
    df = pd.concat([m_df, v_df])
    del m_df
    del v_df
    df = df.reset_index(drop=True)
    df = df[df['nrtracks'] > 10]
    df = df[df['frame'] < 191]
    df = add_surface_variables(df, threshold=threshold, thresh_col=thresh_col, zs=zs, ys=ys)
    out_df = {
        'path' : [], 
        'phase' : [], 
        'time bin' : [],
        'treatment' : [], 
        'location' : [], 
        'local density' : [], 
        'centre distance' : [], 
        'corrected calcium' : [], 
        'platelet count' : [], 
        'dv (um/s)' : [], 
        'number lost' : [], 
        'number gained' : [], 
        'percentage lost (%)' : [], 
        'percentage gained (%)' : []
    }
    paths = pd.unique(df['path'])
    surface_vars = pd.unique(df['surface'])
    ant_surface_vars = pd.unique(df['anterior_surface'])
    its = len(paths) * len(time_bins)
    # get average results for each injury
    with tqdm(desc='Generating MIPS binned data', total=its) as progress:
        for p in paths:
            pdf = df[df['path'] == p]
            # get average results for each time bin
            for bin in time_bins.keys():
                bmin, bmax = time_bins[bin]
                bdf = pdf[(pdf['time (s)'] > bmin) & (pdf['time (s)'] < bmax)]
                tx = bdf['treatment'].values[0]
                # get results for surface and core
                out_df = get_values_and_generate_row(surface_vars, 'surface', out_df, bdf, tx, bmin, bmax, bin, p)
                # get results for anterior surface and core/tail
                out_df = get_values_and_generate_row(ant_surface_vars, 'anterior_surface', out_df, bdf, tx, bmin, bmax, bin, p)
                progress.update(1)
    bins = [time_bins[key] for key in time_bins.keys()]
    save_name = f'{save_name}_bins-{bins}_thresh-{threshold}_zs-{zs}_ys{ys}.csv'
    save_path = os.path.join(save_dir, save_name)
    out_df = pd.DataFrame(out_df)
    out_df.to_csv(save_path)
    return out_df



def get_values_and_generate_row(position_vars, position_col, out_df, bdf, tx, bmin, bmax, bin, p):
    for var in position_vars:
        vdf = bdf[bdf[position_col] == var]
        # calculate things
        ld = vdf['nb_density_15'].mean()
        cd = vdf['dist_c'].mean()
        cc = vdf['ca_corr'].mean()
        pc = len(pd.unique(vdf['particle']))
        dv = vdf['dv'].mean()
        vdf['terminating'] = vdf['tracknr'] == vdf['nrtracks']
        nldf = vdf[vdf['terminating'] == True]
        nl = len(nldf)
        del nldf
        frames = pd.unique(vdf['frame'])
        ngs = []
        for f in frames:
            fdf = vdf[vdf['frame'] == f]
            ids_0 = pd.unique(fdf['particle'])
            fdf = vdf[vdf['frame'] == f+1]
            if len(fdf) > 0:
                ids_1 = pd.unique(fdf['particle'])
                new = [pid for pid in ids_1 if pid not in ids_0]
                ngs.append(len(new))
            else:
                ngs.append(np.NaN)
        ng = np.nansum(ngs)
        if pc > 0:
            pl = (nl / pc) * 100
            pg = (ng / pc) * 100
        else:
            pl = 0
            pg = 0

        # append things
        out_df = generate_row(out_df, p, bin, bmin, bmax, tx, var, ld, cd, cc, pc, dv, nl, ng, pl, pg)
    return out_df
 


def generate_row(out_df, p, bin, bmin, bmax, tx, var, ld, cd, cc, pc, dv, nl, ng, pl, pg):
    out_df['path'].append(p)
    out_df['phase'].append(bin)
    out_df['time bin'].append(f'{bmin}-{bmax} s')
    out_df['treatment'].append(tx)
    out_df['location'].append(var)
    out_df['local density'].append(ld)
    out_df['centre distance'].append(cd)
    out_df['corrected calcium'].append(cc)
    out_df['platelet count'].append(pc)
    out_df['dv (um/s)'].append(dv)
    out_df['number lost'].append(nl)
    out_df['number gained'].append(ng)
    out_df['percentage lost (%)'].append(pl)
    out_df['percentage gained (%)'].append(pg)
    return out_df



def box_plots_for_mips(df, variable, phase='consolidation phase', x='location', hue='treatment'):
    sdf = df[df['phase'] == phase]
    print(len(sdf), len(df))
    sns.boxplot(x=x, y=variable, data=sdf, hue=hue)
    plt.show()



def bar_plots_for_mips(df, variable, phase='transition phase', x='location', hue='treatment'):
    sdf = df[df['phase'] == phase]
    sdf = sdf.dropna(subset=[variable, ])
    vars = pd.unique(df['location'])
    for v in vars:
        print(f'T test: {variable} in {v} - MIPS vs vehicle during {phase}')
        vdf = sdf[sdf['location'] == v]
        mdf = vdf[vdf['treatment'] == 'MIPS']
        m = mdf[variable].values
        cdf = vdf[vdf['treatment'] == 'vehicle ']
        c = cdf[variable].values
        result = stats.ttest_ind(m, c) # need to do a two way anova and look at interaction effects between treatment and location
        print(result)
    sns.barplot(x=x, y=variable, data=sdf, hue=hue)
    plt.show()



def two_way_anova_for_mips(df, vars, two_groups=('surface', 'core'), loc_col='location', tx_col='treatment'):
    df0 = df[df[loc_col] == two_groups[0]]
    df1 = df[df[loc_col] == two_groups[1]]
    df = pd.concat([df0, df1])
    df = df.reset_index(drop=True)
    for v in vars:
        lm = f'Q("{v}") ~ {loc_col} + {tx_col} + {loc_col}:{tx_col}'
        #lm = f'Q({v}) ~ Q({loc_col}) + Q({tx_col}) + Q({loc_col}):Q({tx_col})'
        #lm = f'Q({v}) ~ C({loc_col}) + C({tx_col}) + C({loc_col}):C({tx_col})'
        model = ols(formula=lm, data=df).fit()
        res = sm.stats.anova_lm(model, typ=2)
        print(f'Two way anova for {v}: location x treatment')
        print(res)



# Some box plots for growth vs consolidation (260 seconds, 180 seconds for letting go of mips surface platelets)
# total distance before release 
# 

if __name__ == '__main__':
    d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
    mips_n = '211206_mips_df_220818.parquet'
    saline_n = '211206_saline_df_220827_amp0.parquet'
    dmso_n = '211206_veh-mips_df_220831.parquet'
    mpath = os.path.join(d, mips_n)
    spath = os.path.join(d, dmso_n)
    #mdf = pd.read_parquet(mpath)
    #sdf = pd.read_parquet(spath)
    #mdf['nrterm'] = mdf['nrtracks'] - mdf['tracknr']
    #sdf['nrterm'] = sdf['nrtracks'] - sdf['tracknr']
    #timeplot_surface_comparison(mdf, sdf, y_col='dv', threshold=50, thresh_col='nb_density_15_pcntf', zs=15, ys=0)
    #timeplot_surface_comparison(mdf, sdf, y_col='ca_corr', threshold=50, thresh_col='nb_density_15_pcntf', zs=15, ys=0)
    #timeplot_surface_comparison(mdf, sdf, y_col='nrterm', threshold=50, thresh_col='nb_density_15_pcntf', zs=15, ys=0)
    #timeplot_surface_comparison(mdf, sdf, y_col='nb_density_15', threshold=70, thresh_col='nb_density_15_pcntf', zs=15, ys=0)
    #timeplot_surface_comparison(mdf, sdf, y_col='dvy', threshold=50, thresh_col='nb_density_15_pcntf', zs=15, ys=0)
    #timeplot_surface_comparison(mdf, sdf, y_col='dist_c', threshold=70, thresh_col='nb_density_15_pcntf', zs=15, ys=0)
    sd = '/Users/amcg0011/Data/platelet-analysis/MIPS_surface'
    sn = 'MIPSvsDMSO'
    #out = generate_mips_data(mdf, sdf, sd, sn)

    #out = generate_mips_data(mdf, sdf, sd, sn, threshold=50)
    #out = generate_mips_data(mdf, sdf, sd, sn, threshold=70)
    #out = generate_mips_data(mdf, sdf, sd, sn, threshold=40)

    sp = '/Users/amcg0011/Data/platelet-analysis/MIPS_surface/MIPSvsDMSO_bins-[(0, 180), (180, 260), (260, 600)]_thresh-40_zs-15_ys0.csv'
    df = pd.read_csv(sp)
    vars = ['local density', 'centre distance', 'corrected calcium', 'platelet count', 
            'dv (um/s)', 'number lost', 'number gained', 'percentage lost (%)', 'percentage gained (%)']
    for v in vars:
        bar_plots_for_mips(df, v)
    #two_way_anova_for_mips(df, vars, two_groups=('surface', 'core'), loc_col='location', tx_col='treatment')
    #two_way_anova_for_mips(df, vars, two_groups=('anterior surface', 'core and posterior'), loc_col='location', tx_col='treatment')

    
