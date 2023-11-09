import pandas as pd
import os
from plateletanalysis.variables.basic import get_treatment_name
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

def plot_scatter(data, x, y, hue='treatment'):
    plt.rcParams['svg.fonttype'] = 'none'
    sns.scatterplot(x=x, y=y, hue=hue, data=data)
    sns.despine()
    plt.show()

def plot_scatter_line(data, x, y, b, m, hue='treatment'):
    plt.rcParams['svg.fonttype'] = 'none'
    sns.scatterplot(x=x, y=y, hue=hue, data=data)
    sns.despine()
    plt.axline((0, b), (1, b + m), linestyle="--")
    plt.show()


def plot_box_plots(data, x, y):
    plt.rcParams['svg.fonttype'] = 'none'
    sns.boxplot(x=x, y=y, data=data, palette='Set2')
    sns.stripplot(data=data, x=x, y=y, dodge=True, edgecolor = 'white', linewidth=0.3, palette='Set2')
    sns.despine()
    plt.show()




sd = '/Users/abigailmcgovern/Data/platelet-analysis/toroidal_paper/data'
names = ['231009_saline', 'bivalirudin', 'cangrelor', 'control', 'DMSO(MIPS)_and_MIPS_2023', 
         'DMSO(MIPS)', 'DMSO(SQ)', 'MIPS', 'PAR4--', 'saline_gavarge_vehicle', 'SQ', 'p-selectin']
paths = [os.path.join(sd, f'{n}_summary_data.csv') for n in names]

#dfs = [pd.read_csv(p) for p in paths]
#dfs = pd.concat(dfs).reset_index(drop=True)

#dfs['treatment'] = dfs['path'].apply(get_treatment_name)



vars = ['mean platelet count -C', 'n embolysis events', 
        'n embolysis events - C', 'recruitment - C', 
        'shedding - C', 'dvz - G', 'dv - C', 
        'instability - C', 'initial decelleration', 
        'dvy - C' , 'dv - G', 'donutness mean', 
        'donutness prominence']


def inhibited(tr):
    if tr in ['cangrelor', 'bivalirudin', 'PAR4--', 'SQ', 'MIPS']:
        return True
    else:
        return False

#dfs['toroidal'] = False
#dfs['toroidal'] = dfs['donutness prominence'] != 0

def non_tor(df):
    sdf = df[df['treatment'] == 'cangrelor']
    sdf = df[df['treatment'] == 'bivalirudin']
    sdf = df[df['treatment'] == 'PAR4--']

sp = os.path.join(sd, f'231012_mega_summary_data.csv')
#dfs.to_csv(sp)
dfs = pd.read_csv(sp)

dfs['inhibited'] = dfs['treatment'].apply(inhibited)

dfs = dfs[dfs['inhibited'] == False]



#plot_box_plots(dfs, 'toroidal', 'dvz - G')
#plot_box_plots(dfs, 'toroidal', 'n embolysis events')
#density
#plot_box_plots(dfs, 'toroidal', 'density')

dfs_t = dfs[dfs['toroidal']]
dfs_n = dfs[dfs['toroidal'] == False]

#res = stats.ttest_ind(dfs_t['n embolysis events - C'].values, dfs_n['n embolysis events - C'].values)
#print(res) # Ttest_indResult(statistic=-3.7874589092392155, pvalue=0.00027441649159005836)

#res = stats.ttest_ind(dfs_t['dvz - G'].values, dfs_n['dvz - G'].values)
#print(res) # Ttest_indResult(statistic=4.017388761583911, pvalue=0.00012185041667879984)

#res = stats.ttest_ind(dfs_t['n embolysis events'].values, dfs_n['n embolysis events'].values)
#print(res) # Ttest_indResult(statistic=-4.19208767253414, pvalue=6.451443274843057e-05)

#res = stats.ttest_ind(dfs_t['density'].values, dfs_n['density'].values)
#print(res) # Ttest_indResult(statistic=2.7485808330735098, pvalue=0.007232097933203161)

x = 'donutness mean'
x = 'latency donutness'
#x = 'donutness prominence'
y = 'mean platelet count -C'
y = 'n embolysis events'
y = 'instability'
y = 'nrtracks'
y = 'dvy - C'
y = 'initial decelleration'
y = 'latency max count'

#dfs = dfs[dfs['treatment'] == 'control']

dfs_t = dfs_t[dfs_t['treatment'] != 'DMSO (SQ)']
dfs_t = dfs_t[dfs_t['treatment'] != 'DMSO (MIPS)']
dfs_t = dfs_t[dfs_t['treatment'] != 'DMSO (salgav)']
#plot_scatter(dfs_t, x, y, hue='treatment')
#res = stats.linregress(dfs_t[x].values, dfs_t[y].values)
#print(res) 

# latencys LinregressResult(slope=0.586687347662309, intercept=127.66883940684338, 
# rvalue=0.45130062247365904, pvalue=0.01592844397672208, stderr=0.22750976426250624, intercept_stderr=29.96209204208317)
plot_scatter_line(dfs_t, x, y, 127.66883940684338, 0.586687347662309, hue='treatment')

#dfs = dfs[dfs['treatment'] != 'saline']
#plot_box_plots(dfs, 'toroidal', y)
#dfs_t = dfs_t[dfs_t['treatment'] != 'saline']
#dfs_n = dfs_n[dfs_n['treatment'] != 'saline']
#res = stats.ttest_ind(dfs_t[y].values, dfs_n[y].values)
#print(res) 

# mean count and donutness
#LinregressResult(slope=163.41557763806884, intercept=999.9964949894825, 
# rvalue=0.14452576773067527, pvalue=0.2924544627597035, stderr=153.68323271507126, intercept_stderr=281.59050053210507)

# emb and donutness
#LinregressResult(slope=-3.0124169207420066, intercept=8.488590693883344, 
# rvalue=-0.38483541948977323, pvalue=0.003718577929477012, stderr=0.9924230649643491, intercept_stderr=1.8183955573151562)

# nrtracks Ttest_indResult(statistic=2.372808734999041, pvalue=0.020217063313134595)

# yvel Ttest_indResult(statistic=4.478649871445015, pvalue=2.1981450959517472e-05)