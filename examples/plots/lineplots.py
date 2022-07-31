import pandas as pd
import os
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from plateletanalysis.variables.measure import quantile_normalise_variables


def add_variable_centile_bins(
        df, 
        var, 
        pcnt_bands=((0, 25), (25, 50), (50, 75), (75, 100)), 
        names=('25th centile', '50th centile', '75th centile', '100th centile')
        ):
    for i, pb in enumerate(pcnt_bands):
        l, u = pb
        n = names[i]
        rdf = df[(df[var] >= l) & (df[var] < u)]
        idxs = rdf.index.values
        df.loc[idxs, 'centile band'] = n
    return df

       


d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
p = os.path.join(d, '211206_saline_df_220614-amp0.parquet')
df = pd.read_parquet(p)

nrterm = df['nrterm'].values
n_remain = df['frame'].max() - df['frame'].values
df['stability'] = nrterm / n_remain

vars = [
    'cont_tot', 'ca_corr', 'rho', 'theta', 'phi', 
    'nb_density_15', 'nrterm', 'disp', 'rho_diff', 'phi_diff', 
    'theta_diff', 'nd15_percentile', 'fibrin_dist', 'fibrin_cont', 
    'phi_pcnt','phi_diff_pcnt', 'rho_pcnt', 'theta_pcnt','zs_pcnt', 
    'fibrin_dist_pcnt', 'nb_ca_corr_15', 'nb_ca_diff_15_diff',
    'nb_ca_copying_15', 'nb_cont_15', 'stability', 'zs', 'dv'
    ]

rho_pcnt_bands = ((0, 25), (25, 50), (50, 75), (75, 100))
names = ('25th centile', '50th centile', '75th centile', '100th centile')

df = quantile_normalise_variables(df, ('nb_cont_15', ))

df = add_variable_centile_bins(df, 'nb_cont_15_pcnt')

df = df.dropna(subset=['centile band', 'stability'])
df = df[df['stability'] < 2]

df['seconds'] = df['frame'] / 0.321764322705706

now = datetime.now()
dt = now.strftime("%y%m%d_%H%M%S")
s = f'/Users/amcg0011/PhD/Platelets/Results/Lineplots/{dt}_nb_cont_15-centile-band'
os.makedirs(s, exist_ok=True)

sns.set(font_scale = 1.5)
sns.set_style("white")
for var in vars:
    sns.lineplot(data=df, x='seconds', y=var, hue='centile band', hue_order=names, palette='rocket', lw=2)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    sns.despine()
    sp = os.path.join(s, f'{var}_centile-band-plot.png')
    plt.savefig(sp, dpi=300)
    plt.close()
       