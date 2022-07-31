from plateletanalysis.analysis.stats import multivariate_regression_model
import pandas as pd
import os

d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
s = os.path.join(d, '211206_saline_df_220613-amp0.parquet')
df = pd.read_parquet(s)

#df = df[df['frame'] > 50]
#df = df[df['fibrin_dist_pcnt'] < 98]
nrterm = df['nrterm'].values
n_remain = df['frame'].max() - df['frame'].values
df['stability'] = nrterm / n_remain
df = df.dropna(subset=['stability'])
df = df[df['stability'] < 2] # only filtering out inf

multivariate_regression_model(df, formula='stability ~ dv')