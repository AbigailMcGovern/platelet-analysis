import pandas as pd
from plateletanalysis.topology.animations import persistance_diagrams_for_timepointz
from plateletanalysis.topology.donutness import scale_x_and_y, donutness_data
from plateletanalysis.variables.measure import quantile_normalise_variables_frame
from plateletanalysis.variables.basic import add_time_seconds
from plateletanalysis.topology.simulations import simulate_matching_data
import os
import seaborn as sns
from collections import defaultdict
import matplotlib.pyplot as plt


def smooth_vars(df, vars, w=20):
    df = df.sort_values('frame')
    for v in vars:
        for k, grp in df.groupby(['path', 'bootstrap_id']):
            rolled = grp[v].rolling(window=w, center=True).mean()
            idxs = grp.index.values
            df.loc[idxs, v] = rolled
    return df


def plot_donutness(data, save_path):
    # Average of bootstrapping
    exp_data = defaultdict(list)
    for k, grp in data.groupby(['path', 'time (s)', 'type']):
        exp_data['path'].append(k[0])
        exp_data['time (s)'].append(k[1])
        exp_data['type'].append(k[2])
        exp_data['donutness'].append(grp['donutness'].mean())
    exp_data = pd.DataFrame(exp_data)
    # Plotting
    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
    sns.lineplot(data=exp_data, x='time (s)', y='donutness', hue='type', ax=axs[0])
    sns.lineplot(data=data[data['type'] == 'thrombi'], x='time (s)', y='donutness', hue='path', ax=axs[1])
    sns.despine(ax=axs[0])
    sns.despine(ax=axs[1])
    fig.set_size_inches(4.2, 6)
    fig.subplots_adjust(right=0.97, left=0.13, bottom=0.13, top=0.97, wspace=0.3, hspace=0.2)
    fig.savefig(save_path)
    plt.show()



p = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/211206_saline_df_spherical-coords.parquet'
demo_path = '200527_IVMTR73_Inj4_saline_exp3'
sd = '/Users/abigailmcgovern/Data/platelet-analysis/toroidal_paper/figure_1'

plt.rcParams['svg.fonttype'] = 'none'

df = pd.read_parquet(p)
#df = scale_x_and_y(df)
y_col_s = 'ys_scaled'
y_col = 'ys'
x_col_s = 'x_s_scaled'
x_col = 'x_s'

#df = quantile_normalise_variables_frame(df, )
#df.to_parquet(p)

tps = (20, 60, 120)

#persistance_diagrams_for_timepointz(df, centile=50, x_col=x_col_s, y_col=y_col_s, units='um', path=demo_path, tps=tps)
#out = donutness_data(df, units='AU')
sp = os.path.join(sd, 'saline_donut_data_scaled_sn200_n100_c50.csv')
#out.to_csv(sp)

#out = pd.read_csv(sp)


#sns.lineplot(data=out, x='frame', y='donutness', hue='path')
#sns.lineplot(data=out, x='frame', y='donutness')
#plt.show()
#out = out[out['frame'] < 10]
#ssp = os.path.join(sd, 'simulated_topologies_saline.csv')
#sim = simulate_matching_data(data=out, save_path=ssp)
#
ssp1 = os.path.join(sd, 'simulated_saline_donutdata_disc.csv')
ssp2 = os.path.join(sd, 'simulated_saline_donutdata_1-loop.csv')
ssp3 = os.path.join(sd, 'simulated_saline_donutdata_2-loop.csv')
#
#
#disc = donutness_data(sim, x_col='x_disc', y_col='y_disc', units='AU', filter_col=None)
#disc.to_csv(ssp1)
#loop1 = donutness_data(sim, x_col='x_1-loop', y_col='y_1-loop', units='AU', filter_col=None)
#loop1.to_csv(ssp2)
#loop2 = donutness_data(sim, x_col='x_2-loop', y_col='y_2-loop', units='AU', filter_col=None)
#loop2.to_csv(ssp3)

#sns.lineplot(data=data, x='frame', y='donutness', hue='type')
#plt.show()

disc = pd.read_csv(ssp1)
disc['type'] = 'disc'
disc =  smooth_vars(disc, ['donutness', ])

loop1 = pd.read_csv(ssp2)
loop1['type'] = '1-loop'
loop1 =  smooth_vars(loop1, ['donutness', ])

loop2 = pd.read_csv(ssp3)
loop2['type'] = '2-loop'
loop2 =  smooth_vars(loop2, ['donutness', ])

sp = os.path.join(sd, 'saline_donut_data_scaled_sn200_n100_c50.csv')
out = pd.read_csv(sp)
out['type'] = 'thrombi'
out = smooth_vars(out, ['donutness', ])

data = pd.concat([disc, loop1, loop2, out]).reset_index(drop=True)
data = add_time_seconds(data)
#data['persistence_2'] = data['death_2'] - data['birth_2'] # fix error in code (correced at source)
#data['persistence_difference'] = data['persistence_1'] - data['persistence_2'] 

save_path = os.path.join(sd, 'saline_persistence_donutness_w_sim_averaged.svg')
plot_donutness(data, save_path)

