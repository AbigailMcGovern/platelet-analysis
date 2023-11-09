import pandas as pd
from plateletanalysis.topology.animations import persistance_diagrams_for_timepointz
from plateletanalysis.topology.donutness import scale_x_and_y, donutness_data
from plateletanalysis.variables.measure import quantile_normalise_variables_frame
from plateletanalysis.variables.basic import add_time_seconds, get_treatment_name
from plateletanalysis.topology.simulations import simulate_matching_data
import os
import seaborn as sns
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import scoreatpercentile

def smooth_vars(df, vars, w=20):
    df = df.sort_values('frame')
    for v in vars:
        for k, grp in df.groupby(['path', 'bootstrap_id']):
            rolled = grp[v].rolling(window=w, center=True).mean()
            idxs = grp.index.values
            df.loc[idxs, v] = rolled
    return df



def plot_donutness(data, save_path, hue='treatment', gb=['path', 'time (s)', 'treatment']):
    # Average of bootstrapping
    data = smooth_vars(data, ['donutness', ], w=20)
    exp_data = defaultdict(list)
    data = data[data['donutness'] != 2.0]
    for k, grp in data.groupby(gb):
        for i, c in enumerate(gb):
            exp_data[c].append(k[i])
        exp_data['donutness'].append(grp['donutness'].mean())
    exp_data = pd.DataFrame(exp_data)
    # Plotting
    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
    sns.lineplot(data=data, x='time (s)', y='donutness', hue=hue, ax=axs[0])
    sns.lineplot(data=data, x='time (s)', y='donutness', hue='path', ax=axs[1])
    sns.despine(ax=axs[0])
    sns.despine(ax=axs[1])
    fig.set_size_inches(4.2, 6)
    fig.subplots_adjust(right=0.97, left=0.13, bottom=0.13, top=0.97, wspace=0.3, hspace=0.2)
    fig.savefig(save_path)
    plt.show()



def classify_exp_type(path):
    if path.find('exp5') != -1:
        return '10-20 min'
    elif path.find('exp3') != -1:
        return '0-10 min'
    else:
        return 'other'


p = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/230919_p-selectin.parquet'
sd = '/Users/abigailmcgovern/Data/platelet-analysis/toroidal_paper/figure_2'
n = 'p-sel_PS-fluor_donutness_trial'
cols = ['Alxa 647: mean_intensity', 'Alxa 647: max_intensity', 
 'GaAsP Alexa 488: mean_intensity', 'GaAsP Alexa 488: max_intensity',
 'GaAsP Alexa 568: mean_intensity', 'GaAsP Alexa 568: max_intensity']
psel_chan = 'GaAsP Alexa 568: mean_intensity'

# Read data
# ---------
df = pd.read_parquet(p)
#print(df.columns.values)
df = df.rename(columns={'GaAsP Alexa 568: mean_intensity' : 'p-sel average intensity'})


# Density Donutness
# -----------------
#sp0 = os.path.join(sd, 'p-sel_density_donutness_data.csv')
df = df[df['nrtracks'] > 1]
#out = donutness_data(df, units='AU')
#out.to_csv(sp0)


# Working out
# -----------
#demo_name = '211021_IVMTR139_Inj3_DMSO_exp3'
#demo_name = pd.unique(df.path)[2]
print(scoreatpercentile(df['p-sel average intensity'].values, 80))
# Score at 95th centile is 543.9049955791336
# Score at 90th centile is 429.32221971576826
# Score at 85th centile is 342.39207161125313
# Score at 80th centile is 293.2623613180991
#print(demo_name)
#df = df[df['path'] == demo_name]
#df['p-sel average intensity'].hist(bins=200)
#plt.yscale('log')
#plt.show()
#sdf = df[(df['frame'] == 60) & (df['p-sel average intensity'] > 300) & (df['p-sel average intensity'] < 800)]
#plt.show()
#plt.scatter(sdf['x_s'].values, sdf['ys'].values)


# Quantify p-sel donutness
# ------------------------
n = '231018_p-sel_PS-fluor_donutness_gt429_90thC.csv'
sp1 = os.path.join(sd, n)
out = donutness_data(df, units='AU', filter_col='p-sel average intensity', centile=429)
out.to_csv(sp1)
#out = pd.read_csv(sp1)
#out = out[out['exp_type'] == '0-10 min']
out = add_time_seconds(out)
out['treatment'] = out['path'].apply(get_treatment_name)
out['exp_type'] = out['path'].apply(classify_exp_type)
for k, grp in out.groupby('exp_type'):
    if k == '10-20 min':
        idxs = grp.index.values
        out.loc[idxs, 'time (s)'] = out.loc[idxs, 'time (s)'] + 600
n = '231018_p-sel_PS-fluor_donutness_gt429_90thC_plot.svg'
sp2 = os.path.join(sd, n)
plot_donutness(out, sp2)


# Read p-sel donut data
# ---------------------
#out = pd.read_csv(sp1)


# Fix frame numbers for long duration exp
# ---------------------------------------
#out['exp_type'] = out['path'].apply(classify_exp_type)
#sout = out[out['exp_type'] == '10-20 min']
#frames = sout['frame'].values + 193
#idxs = sout.index.values
#out.loc[idxs, 'frame'] = frames


# graph all
# ---------
#out = smooth_vars(out, vars=('donutness', ))
#sns.lineplot(data=out, x='frame', y='donutness', hue='path')
#plt.show()



