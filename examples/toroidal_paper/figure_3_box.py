import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from plateletanalysis.variables.basic import get_treatment_name, time_seconds, add_region_category


def boxplot_for_var(data, var, order=['saline', 'bivalirudin', 'PAR4--']):
    data = data.groupby(['path', 'particle', 'treatment'])[var].mean().reset_index()
    data = data.groupby(['path', 'treatment'])[var].mean().reset_index()
    fig, ax = plt.subplots(1, 1)
    # stats
    for k, grp in data.groupby('treatment'):
        vals = grp[var].values
        print(f'{k} mean and SEM')
        print(vals.mean(), vals.std() / np.sqrt(len(vals)))
        print(f'{k} 95% CI')
        print(stats.t.interval(alpha=0.95, df=len(vals)-1, loc=np.mean(vals), scale=stats.sem(vals)) )
    # plot
    plt.rcParams['svg.fonttype'] = 'none'
    sns.boxplot(data=data, x='treatment', y=var, ax=ax, palette='rocket', order=order)
    sns.stripplot(data=data, x='treatment', y=var, order=order,
                  ax=ax, palette='rocket', dodge=True, edgecolor = 'white', linewidth=0.3, jitter=True, size=6)
    sns.despine(ax=ax)
    fig.set_size_inches(3, 3)
    fig.subplots_adjust(right=0.97, left=0.23, bottom=0.17, top=0.9, wspace=0.4, hspace=0.4)
    plt.show()


p0 = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/211206_biva_df.parquet'
p1 = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/211206_par4--_df.parquet'
p2 = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/211206_saline_df_spherical-coords.parquet'

ps = [p0, p1, p2]

df = [pd.read_parquet(p) for p in ps]
df = pd.concat(df).reset_index(drop=True)

df['treatment'] = df['path'].apply(get_treatment_name)
df = df[df['nrtracks'] > 1]
df = df.rename(columns={'c1_mean' : 'fibrin intensity (AU)'})
df = time_seconds(df)
df = add_region_category(df)
df = df[df['time (s)'] > 300]
df = df[df['region'] == 'center']


#boxplot_for_var(df, 'fibrin intensity (AU)')

data = df.groupby(['path', 'particle', 'treatment'])['fibrin intensity (AU)'].mean().reset_index()
data = data.groupby(['path', 'treatment'])['fibrin intensity (AU)'].mean().reset_index()
arrs = {k : grp['fibrin intensity (AU)'] for k, grp in data.groupby('treatment')}
for k in arrs:
    print(k, len(arrs[k]))
#res = stats.mannwhitneyu(arrs['saline'], arrs['bivalirudin'])
#print(res)
res = stats.mannwhitneyu(arrs['saline'], arrs['PAR4--'])
print(res)

# -------------------
# Statistical results
# -------------------
# saline: n = 15
# biva: n = 17
# par4--: n = 20

# all regions > 300 s
# -------------------
# PAR4-- mean and SEM
#305.9700909939701 80.01139047296884
#PAR4-- 95% CI
#(134.15384150496106, 477.7863404829791)
#bivalirudin mean and SEM
#169.6180107544617 1.3691377534510223
#bivalirudin 95% CI
#(166.62624162933875, 172.60977987958464)
#saline mean and SEM
#199.1618544139165 5.614801499873135
#saline 95% CI
#(186.69662934612865, 211.62707948170436)


# centre > 300 s
# --------------
#PAR4-- mean and SEM
#470.6706833647028 154.17306960133033
#PAR4-- 95% CI
#(139.59983914461952, 801.7415275847861)
#bivalirudin mean and SEM
#162.38033778404196 1.077258852688304
#bivalirudin 95% CI
#(160.02636736206316, 164.73430820602076)
#saline mean and SEM
#204.87823730028506 5.187043794353324
#saline 95% CI
#(193.36266220211866, 216.39381239845147)
# saline vs biva - Mann Whitney U test
# MannwhitneyuResult(statistic=255.0, pvalue=1.6197135918578049e-06)
# saline vs par4 - Mann Whitney U test
#MannwhitneyuResult(statistic=73.0, pvalue=0.010772291908133374)