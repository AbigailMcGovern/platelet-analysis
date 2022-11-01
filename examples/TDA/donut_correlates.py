import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from plateletanalysis.topology.largest_loop import find_max_donut_time
import numpy as np



def find_donut_correlates(data, donut_desc, donut_var, data_var, data_var_frames, sp=None):
    data0 = data[(data['frame'] >= data_var_frames[0]) & (data['frame'] < data_var_frames[1])]
    #data_gb = data0.groupby(['path'])
    t = f'{data_var} mean for frames {data_var_frames}'
    paths = donut_desc['path'].values
    df = {
        'path' : paths,
        'treatment' : list(donut_desc['treatment'].values), 
        donut_var : list(donut_desc[donut_var].values), 
        data_var : [np.nanmean(data0[data0['path'] == p][data_var].values) for p in paths]
    }
    df = pd.DataFrame(df)
    #print(df.head())
    if sp is not None:
        df.to_csv(sp)
    ax = sns.scatterplot(x=donut_var, y=data_var, hue='treatment', data=df)
    ax.set_title(t)
    plt.show()




if __name__ == '__main__':
    sd = '/Users/amcg0011/Data/platelet-analysis/TDA/treatment_comparison'
    save_path = os.path.join(sd, '221025_longest-loop-analysis.csv')
    data = pd.read_csv(save_path)
    donut_desc, summary = find_max_donut_time(data, '221025_longest-loop-analysis_donut-desc', sd)
    print(donut_desc.head())
    print(donut_desc.columns.values)
    donut_vars = ['max difference from mean (std dev) rolling', 'time (s)', 'time over 6']
    #data_vars = ['core count', 'surface count', 'core turnover', 'surface turnover', 'core lost', 'surface lost', 'core gained', 'surface gained']
    #data_vars = ['core to surface', 'surface to core', 'core gained (%)', 'core lost (%)']
    #data_vars = ['core turnover (%)', 'surface turnover (%)', 'tail turnover (%)']
    data['core to surface (%)'] = data['core to surface'] / data['core count'] * 100
    data['surface to core (%)'] = data['surface to core'] / data['surface count'] * 100
    data['core to tail (%)'] = data['core to tail'] / data['core count'] * 100
    data_vars = ['core to surface (%)', 'surface to core (%)', 'core to tail (%)']
    frames = [(0, 50), (50, 100), (100, 190), ]
    for dnv in donut_vars:
        for dav in (data_vars):
            for f in frames:
                find_donut_correlates(data, donut_desc, dnv, dav, f)


#['pid', 'index', 'path', 'frame', 'x_s', 'ys', 'zs', 'c0_mean', 'c0_max', 'c1_mean', 'c1_max', 'c2_mean', 'c2_max', 'vol', 'elong',

 #'flatness', 'treatment', 'cohort', 'eigval_0', 'eigval_1', 'eigval_2', 'zf', 'stab', 'nba_d_5', 'nba_d_10', 'nba_d_15', 'particle',
 
  #'cont_p', 'depth', 'tracknr', 'nrtracks', 'cont_tot', 'displ_tot', 'dvz_tot', 'inh', 'inj', 'date', 'mouse', 'time', 'minute', 'tracked', 
  
  #'dist_c', 'dist_cz', 'exp_id', 'inh_exp_id', 'dvz_s', 'cont_s', 'mov_class', 'movement', 'position', 'inside_injury', 'height', 'z_pos', 
  
  #'zz', 'zled', 'ca_corr', 'nb_particles_15', 'nb_disp_15', 'nb_density_15', 'nb_density_15_pcnt', 'nb_density_15_pcntf', 'x_s_pcnt', 
  
  #'ys_pcnt', 'dist_c_pcntf', 'x_s_orig', 'ys_orig', 'zs_orig', 'rho', 'theta', 'phi', 'dvx', 'dvy', 'dvz', 'dv', 'cont', 'cont_p',
  
   #'phi_diff', 'theta_diff', 'rho_diff']