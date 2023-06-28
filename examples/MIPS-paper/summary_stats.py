import pandas as pd

save_path = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/Figure_4/230523_exp_region_phase_plot_data.csv'
result = pd.read_csv(save_path)

result = result[result['phase'] == 'consolidation']
dmso = result[result['treatment'] == 'DMSO (MIPS)']
mips = result[result['treatment'] == 'MIPS']

regions = ('center', 'anterior', 'lateral', 'posterior')
for r in regions:
    print(r)
    dmso_c = dmso[dmso['region'] == r]['platelet count']
    dmso_r = dmso[dmso['region'] == r]['recruitment']
    dmso_p = dmso_r / dmso_c * 100
    print('DMSO count:', dmso_c.mean(), ' +/- ',  dmso_c.sem())
    print('DMSO recruitment:', dmso_r.mean(), ' +/- ',  dmso_r.sem())
    print('DMSO percentage:', dmso_p.mean(), ' +/- ',  dmso_p.sem())
    mips_c = mips[mips['region'] == r]['platelet count']
    mips_r = mips[mips['region'] == r]['recruitment']
    mips_p = mips_r / mips_c * 100
    print('MIPS count:', mips_c.mean(), ' +/- ',  mips_c.sem())
    print('MIPS recruitment:', mips_r.mean(), ' +/- ',  mips_r.sem())
    print('MIPS percentage:', mips_p.mean(), ' +/- ',  mips_p.sem())
    mw = stats.mannwhitneyu(dmso_c, mips_c)
    print('count: ', mw)
    mw = stats.mannwhitneyu(dmso_r, mips_r)
    print('recruitment: ', mw)
    mw = stats.mannwhitneyu(dmso_p, mips_p)
    print('pcnt: ', mw)


for r in regions:
    print(r)
    dmso_l = dmso[dmso['region'] == r]['average net platelet loss (/min)']
    print('DMSO loss:', dmso_l.mean(), ' +/- ',  dmso_l.sem())
    mips_l = mips[mips['region'] == r]['average net platelet loss (/min)']
    print('MIPS loss:', mips_l.mean(), ' +/- ',  mips_l.sem())
    

from scipy import stats
for r in regions:
    print(r)
    dmso_l = dmso[dmso['region'] == r]['average net platelet loss (/min)']
    print('DMSO loss:', dmso_l.mean(), ' +/- ',  dmso_l.sem())
    mips_l = mips[mips['region'] == r]['average net platelet loss (/min)']
    print('MIPS loss:', mips_l.mean(), ' +/- ',  mips_l.sem())
    mw = stats.mannwhitneyu(dmso_l, mips_l)
    print(mw)

import numpy as np
mips_l = mips[mips['region'] == 'posterior']['average net platelet loss (/min)']
stats.ttest_1samp(mips_l, 0)
stats.t.interval(alpha=0.95, df=len(mips_l)-1, loc=np.mean(mips_l), scale=stats.sem(mips_l))