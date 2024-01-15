import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from plateletanalysis.variables.basic import get_treatment_name, time_seconds
from plateletanalysis.variables.transform import cylindrical_coordinates
from plateletanalysis.analysis.peaks_analysis import smooth_vars
from scipy import stats
from toolz import curry


# ---------
# Functions
# ---------


def var_at_cylr_donutness_scatter(
        data, 
        ddf,
        var='c1_mean',
        donut_var="donutness magnetude",
        other_var="donutness duration (s)"#"mean platelet count"
        ):
    #data = df[df['time (s)'] > 300]
    #data = data.dropna(subset=[var, 'cyl_r_bin'])
    #print(data[var].mean())
    #data = data.groupby(['path', 'cyl_r_bin'])[var].mean().reset_index()
    #print(len(pd.unique(data.path)))
    #print(data[var].mean())
    # get peaks
    #data = data.sort_values('cyl_r_bin')
    max_dist_func = max_distance(var)
    data = data.groupby('path').apply(max_dist_func).reset_index()
    #data = data.groupby('path')[var].mean().reset_index()
    data = data.set_index('path')
    # donutness
    ddf = ddf.set_index('path')
    ddf = ddf[[donut_var, other_var, "mean platelet count"]]
    # concat
    data = pd.concat([data, ddf], axis=1).reset_index()
    print(len(pd.unique(data.path)))
    data = data.dropna(subset=[donut_var, other_var, var, "mean platelet count"])
    print(len(pd.unique(data.path)))
    # STATS
    donut = data[donut_var].values
    dist = data['distance from centre (um)'].values
    pselpcnt = data[var].values
    res0 = stats.linregress(donut, dist)
    res0_1 = stats.pearsonr(donut, dist, )
    print(f'fib x distance from centre (um)\n', res0, '\n', res0_1)
    res1 = stats.linregress(donut, pselpcnt)
    res1_1 = stats.pearsonr(donut, pselpcnt)
    print(f'fib x {donut_var}', res1, '\n', res1_1)
    # PLOT
    plt.rcParams['svg.fonttype'] = 'none'
    fig, axs = plt.subplots(1, 2,) #len(bin_order_0)) 
    sns.scatterplot(data=data, x=donut, y='distance from centre (um)', palette='rocket', 
                    hue="mean platelet count", ax=axs[0],alpha=0.9)
    sns.scatterplot(data=data, x=donut_var, y=var, palette='rocket', 
                    hue="mean platelet count", ax=axs[1],alpha=0.9)
    sns.despine(ax=axs[0])
    sns.despine(ax=axs[1])
    plot_line(res0, donut, axs[0])
    plot_line(res1, donut, axs[1])
    fig.set_size_inches(5, 3)
    fig.subplots_adjust(right=0.97, left=0.17, bottom=0.15, top=0.9, wspace=0.45, hspace=0.4)
    plt.show()


def plot_line(res, xvar, ax):
    xmin = 0
    xstd = (xvar.max() - xmin) / 100000 #xvar.std()
    xnext = xmin + xstd
    m = res[0]
    b = res[1]
    ymin = xmin * m + b
    ynext =  xnext * m + b
    ax.axline((xmin, ymin), (xnext, ynext), color="grey", linestyle="--")



def add_binns(
        df, 
        lbs=[0, 120, 300, 600], 
        ubs=[120, 300, 600, 1200], 
        lbs_cyl=None, 
        ubs_cyl=None
        ):
    t_bin = time_bin(lbs, ubs)
    df['time_bin'] = df['time (s)'].apply(t_bin)
    bins = ((1, 15), (15, 30), (30, 194))
    bin_func = bin_by_trk(bins)
    df['track_bin'] = df['nrtracks'].apply(bin_func)
    #df = cylindrical_coordinates(df)
    if lbs_cyl is None or ubs_cyl is None:
        lbs_cyl = np.linspace(0, 100)[0:-1]
        ubs_cyl = np.linspace(0, 100)[1:]
    c_bin = cyl_bin(lbs_cyl, ubs_cyl)
    df['cyl_r_bin'] = df['dist_c'].apply(c_bin)
    bins_1 = ((1, 15), (15, 30), (30, 194))
    df['trackn_bin'] = df['tracknr'].apply(bin_func)
    return df


@curry
def get_var_data(x, y, grp):
    out = pd.DataFrame({
        x : [grp[x].mean(), ], 
        y : [np.nanmean(grp[y].values), ], 
        'nrtracks' : [grp['nrtracks'].mean(), ]
    })
    return out

@curry
def time_bin(lbs, ubs, t):
    #lbs = [0, 60, 120, 300, 600]
    #ubs = [60, 120, 300, 600, 1200]
    #lbs = [0, 30, 300]
    #ubs = [30, 300, 1200]
    for l, u in zip(lbs, ubs):
        if t >= l and t < u:
            return f'{l}-{u} s'
@curry    
def cyl_bin(lbs, ubs, t):
    #lbs = np.linspace(0, 100, 25)[0:-1]
    #ubs = np.linspace(0, 100, 25)[1:]
    for l, u in zip(lbs, ubs):
        if t >= l and t < u:
            return l + 0.5 * (u - l)
        
@curry
def bin_by_trk(bins, val):
    for l, u in bins:
        if val > l and val <= u:
            n = f"{l}-{u} s"
            return n

@curry
def max_distance(var, grp):
    vals = grp[var].values
    max_idx = np.nanargmax(vals)
    mean_val = np.nanmean(vals)#vals[max_idx]
    max_cyl = grp['cyl_r_bin'].values[max_idx]
    out = pd.DataFrame({
        var : [mean_val, ], 
        'distance from centre (um)' : [max_cyl, ]
    })
    return out


# ---------
# Read Data
# ---------

d = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/'
files = ['211206_ctrl_df.parquet', '211206_saline_df_spherical-coords.parquet']
ps = [os.path.join(d, f) for f in files]
dfs = [pd.read_parquet(p) for p in ps]
df = []
for d in dfs:
    print(d.columns.values)
    print(d.c1_mean.mean())
    d = time_seconds(d)
    d = add_binns(d)
    d = d.dropna(subset=['cyl_r_bin', ])
    d = d[d['time (s)'] > 300]
    d = d.groupby(['path', 'cyl_r_bin'])['c1_mean'].mean().reset_index()
    #d = smooth_vars(d, ['c1_mean', ], w=3, t='cyl_r_bin', gb='path')
    #print(d['cyl_r_bin'].value_counts())
    df.append(d)
del dfs
df = pd.concat(df).reset_index()
#df = df.rename(columns={'c1_mean' : 'fibrin average intensity'})
#df = time_seconds(df)
#df = add_binns(df)
#print(pd.unique(df.path))

d = '/Users/abigailmcgovern/Data/platelet-analysis/toroidal_paper/data'
ps0 = [os.path.join(d, 'control_summary_data.csv'), os.path.join(d, 'saline_summary_data_gt1tk.csv')]
ddf = [pd.read_csv(p) for p in ps0]
ddf = pd.concat(ddf).reset_index()
#print(pd.unique(ddf.path))
#ddf = ddf[ddf['donutness duration (s)'] !=0]

# -------
# Execute
# -------

var_at_cylr_donutness_scatter(df, ddf)

# all saline
#  LinregressResult(slope=-0.03992203534843429, intercept=318.69031927270584, rvalue=-0.13632816963147323, 
# pvalue=0.6280571263313381, stderr=0.08046030208480667, intercept_stderr=37.59399568930562) 
# PearsonRResult(statistic=-0.13632816963147326, pvalue=0.6280571263313383)
# fib x donutness magnetude LinregressResult(slope=-11.219810464380757, intercept=334.4995727956156, 
# rvalue=-0.22091221077262138, pvalue=0.42881964018467467, stderr=13.738190680382568, intercept_stderr=42.82897378329387) 
# PearsonRResult(statistic=-0.13632816963147326, pvalue=0.6280571263313383)

# only donut
# fib x donutness duration (s)
# LinregressResult(slope=-0.8308751911563657, intercept=716.6846535863672, rvalue=-0.5490279703090706, 
# pvalue=0.05198788481006804, stderr=0.3813725337514403, intercept_stderr=191.40804789464758) 
# PearsonRResult(statistic=-0.5490279703090706, pvalue=0.05198788481006815)
# fib x donutness magnetude LinregressResult(slope=-29.275983392429488, intercept=396.487044576922, 
# rvalue=-0.3389549078113837, pvalue=0.25724490957164725, stderr=24.500319130979097, intercept_stderr=81.59534924858441) 
# PearsonRResult(statistic=-0.5490279703090706, pvalue=0.05198788481006815)

# saline + ctrl
# fib x donutness duration (s)
# LinregressResult(slope=0.006902507915881121, intercept=195.59043661321462, rvalue=0.08312568332093899, 
# pvalue=0.6349714502779782, stderr=0.014404860135446293, intercept_stderr=5.227633103163378) 
# PearsonRResult(statistic=0.08312568332093888, pvalue=0.6349714502779781)
# fib x donutness magnetude LinregressResult(slope=2.067098264144705, intercept=192.90484920081812, 
# rvalue=0.12551195308851917, pvalue=0.47249817897769997, stderr=2.8442715633028466, intercept_stderr=7.165606526904473) 
# PearsonRResult(statistic=0.08312568332093888, pvalue=0.6349714502779781)

# fib x donutness duration (s)
#  LinregressResult(slope=0.0033073804390311426, intercept=197.27839877738927, rvalue=0.0164257506140068, pvalue=0.9421634446960081, stderr=0.045017910714674286, intercept_stderr=20.606492284584167) 
#  PearsonRResult(statistic=0.01642575061400681, pvalue=0.9421634446960082)
# fib x donutness magnetude LinregressResult(slope=4.443890946515537, intercept=185.27575125761803, rvalue=0.14196966592223895, pvalue=0.5285427477509301, stderr=6.928375853368034, intercept_stderr=21.49902683924279) 
#  PearsonRResult(statistic=0.01642575061400681, pvalue=0.9421634446960082)