import os
import pandas as pd
from scipy import stats
from plateletanalysis.variables.basic import get_treatment_name
from plateletanalysis.analysis.peaks_analysis import groupby_summary_data_mean

d = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/'
ns = ['211206_saline_df_toroidal-coords-1.parquet', '211206_biva_df_toroidal-coords.parquet', 
          '211206_par4--_df_toroidal-coords.parquet']
ps = [os.path.join(d, n) for n in ns]
df = [pd.read_parquet(p) for p in ps]
df = pd.concat(df).reset_index(drop=True)
df['treatment'] = df.path.apply(get_treatment_name)

biva = df[df['treatment'] == 'bivalirudin']
par4 = df[df['treatment'] == 'PAR4--']
saline = df[df['treatment'] == 'saline']
var = 'c1_mean'

biva = groupby_summary_data_mean(biva, 'path', [var, ])
par4 = groupby_summary_data_mean(par4, 'path', [var, ])
saline = groupby_summary_data_mean(saline, 'path', [var, ])

print('saline fibrin:')
print(saline[var].mean(), ' +/- ', saline[var].sem())

print('bivalirudin fibrin:')
print(biva[var].mean(), ' +/- ', biva[var].sem())

res = stats.mannwhitneyu(biva[var].values, saline[var].values)
print(res)

print('PAR4 -/- fibrin:')
print(par4[var].mean(), ' +/- ', par4[var].sem())

res = stats.mannwhitneyu(par4[var].values, saline[var].values)
print(res)


#saline fibrin:
#203.98385084502206  +/-  4.976676195331574
#bivalirudin fibrin:
#168.00701121545256  +/-  1.4648879027597554
#MannwhitneyuResult(statistic=5.0, pvalue=4.084120751399395e-06)
#PAR4 -/- fibrin:
#359.4591571331254  +/-  61.50884808373336
#MannwhitneyuResult(statistic=279.0, pvalue=1.8411402444226534e-05)