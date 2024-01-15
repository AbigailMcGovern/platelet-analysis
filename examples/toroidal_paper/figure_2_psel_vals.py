import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from plateletanalysis.variables.basic import get_treatment_name, adjust_time_for_exp_type, time_seconds, inside_injury_var, cyl_bin
from plateletanalysis.variables.transform import spherical_coordinates, cylindrical_coordinates
from plateletanalysis.analysis.peaks_analysis import smooth_vars, percent_psel_pos
from collections import defaultdict
from scipy import stats


# ---------
# Functions
# ---------

def psel_over_time(df, psel, t):
    plt.rcParams['svg.fonttype'] = 'none'
    df = smooth_vars(df, [psel, ], gb='path' )
    df = df[df['time (s)'] < 600]
    sns.lineplot(data=df, x=t, y=psel, hue='path')
    sns.despine()
    plt.show()


def psel_pcnt_over_time(df, psel_pcnt, t):
    plt.rcParams['svg.fonttype'] = 'none'
    df = df[df['time (s)'] < 600]
    gb = ['path', 'time (s)']
    cols = [psel_pcnt, ]
    #data = groupby_summary_data_mean(df, gb, cols)
    data = percent_psel_pos(df)
    print(data.head())
    data = smooth_vars(data, [psel_pcnt, ], gb='path' )
    sns.lineplot(data=data, x=t, y=psel_pcnt, hue='path')
    sns.despine()
    plt.show()


def psel_f_vs_count(df, psel, count):
    df = df[(df['time (s)'] > 300) & (df['time (s)'] < 600)]
    data = groupby_summary_data_mean(df, ['path', ], [psel, count])
    print(data.head())
    sns.scatterplot(data=data, y=psel, x=count, palette='rocket')
    print(stats.linregress(data[count].values, data[psel].values))
    sns.despine()
    plt.show()


def psel_f_vs_dens_inside(df, dens='nb_density_15', psel='p-sel average intensity'):
    df = df[(df['time (s)'] > 300) & (df['time (s)'] < 600)]
    #df = df[df['zs'] > 15]
    df = df[df['rho'] >= 37.5]
    data = groupby_summary_data_mean(df, ['path', 'time (s)'], [psel, dens])
    data = groupby_summary_data_mean(df, ['path', ], [psel, dens])
    print(data.head())
    sns.scatterplot(data=data, y=psel, x=dens, palette='rocket')
    print(stats.linregress(data[dens].values, data[psel].values))
    sns.despine()
    plt.show()


def psel_f_vs_dens_inside(df, dens='nb_density_15', psel='p-sel average intensity'):
    df = df[(df['time (s)'] > 300) & (df['time (s)'] < 600)]
    #df = df[df['zs'] > 15]
    #df = df[df['rho'] >= 37.5]

    data = groupby_summary_data_mean(df, ['path', 'time (s)'], [psel, dens])
    data = groupby_summary_data_mean(df, ['path', ], [psel, dens])
    print(data.head())
    sns.scatterplot(data=data, y=psel, x=dens, palette='rocket')
    print(stats.linregress(data[dens].values, data[psel].values))
    sns.despine()
    plt.show()

def psel_f_vs_dens_cylrad(df, dens='nb_density_15', psel='p-sel average intensity', cyl='distance from centre (um)'):
    df = df[(df['time (s)'] > 300) & (df['time (s)'] < 600)]
    data = groupby_summary_data_mean(df, ['path', cyl], [psel, dens])
    #data = groupby_summary_data_mean(df, ['path', ], [psel, dens])
    print(data.head())
    plt.rcParams['svg.fonttype'] = 'none'
    sns.scatterplot(data=data, y=psel, x=dens, palette='rocket', hue=cyl)
    print(stats.linregress(data[dens].values, data[psel].values))
    sns.despine()
    plt.show()


def groupby_summary_data_mean(df, gb, cols):
    data = defaultdict(list)
    for k, grp in df.groupby(gb):
        for i, c in enumerate(gb):
            data[c].append(k[i])
        for c in cols:
            data[c].append(grp[c].mean())
    data = pd.DataFrame(data)
    return data

# -------
# Execute
# -------


if __name__ == "__main__":
    do = 'percent'#'average' #'percent
    #do = 'average'

    if do == 'average':
        p = '/Users/abigailmcgovern/Data/platelet-analysis/toroidal_paper/figure_2/230930_p-sel_growth_and_summary_data.csv'
        df = pd.read_csv(p)
        psel = 'P-selectin fluorescence'
        cal = 'Calcium fluorescence'
        dens = 'average density'
        count = 'platelet count'
        yvel = 'average y-axis velocity'
        t = 'time (s)'
        df['treatment'] = df['path'].apply(get_treatment_name)
        #df['exp_type'] = df['path'].apply(classify_exp_type)
        df = df[df['treatment'] == 'control']
        df = adjust_time_for_exp_type(df)
        #psel_over_time(df, psel, t)
        psel_f_vs_count(df, psel, dens)


    elif do == 'percent':
        p = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/230919_p-selectin.parquet'
        df = pd.read_parquet(p)
        df = df[df['nrtracks'] > 1]
        df = time_seconds(df)
        df = df.rename(columns={'GaAsP Alexa 568: mean_intensity' : 'p-sel average intensity'})
        df['treatment'] = df['path'].apply(get_treatment_name)
        df = df[df['treatment'] == 'control']
        df = spherical_coordinates(df)
        df = cylindrical_coordinates(df)
        df = adjust_time_for_exp_type(df)
        df = inside_injury_var(df)
        df['distance from centre (um)'] = df['cyl_radial'].apply(cyl_bin)
        psel_pcnt = 'p-selectin positive platelets (%)'
        t = 'time (s)'
        #psel_pcnt_over_time(df, psel_pcnt, t)
        #psel_f_vs_dens_inside(df)
        psel_f_vs_dens_cylrad(df)


    # PSEL x DENS
    # LinregressResult(slope=-138870.56663524854, intercept=430.00679558778654, 
        # rvalue=-0.6538431044403418, pvalue=0.02909609384613796, stderr=53567.213066516204, 
        # intercept_stderr=61.323728758413765) 
    # PSEL x DENS - centre
    # LinregressResult(slope=-169885.40317261466, intercept=487.9627468775815, 
        # rvalue=-0.5229850251848845, pvalue=0.09879250563283139, stderr=92291.00596290892, 
        # intercept_stderr=109.97939486830475)

