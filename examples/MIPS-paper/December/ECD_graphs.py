from plateletanalysis.variables.basic import time_seconds, get_treatment_name, size_var
import pandas as pd
import os
import seaborn as sns
from scipy import stats
from collections import defaultdict
import matplotlib.pyplot as plt

def cumulative_density(df):
    df = time_seconds(df)
    df = df[df['nrtracks'] > 1]
    df = size_var(df)
    df = df[df['time (s)'] > 260]
    df['treatment'] = df.path.apply(get_treatment_name)
    data = defaultdict(list)
    for k, grp in df.groupby(['path', 'size']):
        n = len(pd.unique(grp['particle']))
        tx = grp['treatment'].values[0]
        data['treatment'].append(tx)
        data['path'].append(k[0])
        data['size'].append(k[1])
        data['count'].append(n)
    data = pd.DataFrame(data)
    #print(data)
    data = data[data['treatment'] != 'DMSO (salgav)']
    data_sml = data[data['size'] == 'small']
    data_lrg = data[data['size'] == 'large']
    print(data_lrg['treatment'].value_counts())
    print(data_sml['treatment'].value_counts())
    fig, axs = plt.subplots(1, 2, sharey=True)
    sns.histplot(x='count', hue='treatment', data=data_sml, element="step", fill=False,
    cumulative=True, stat="density", common_norm=False, ax=axs[0])
    sns.histplot(x='count', hue='treatment', data=data_lrg, element="step", fill=False,
    cumulative=True, stat="density", common_norm=False, ax=axs[1])
    plt.show()



if __name__ == '__main__':
    from plateletanalysis.variables.basic import get_treatment_name
    d = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes'
    from datetime import datetime
    now = datetime.now()
    date = now.strftime("%y%m%d")

    file_names = ('211206_mips_df.parquet', '211206_veh-mips_df.parquet', 
                #  '211206_cang_df.parquet', '211206_saline_df_220827_amp0.parquet', 
                #  '211206_sq_df.parquet', '211206_veh-sq_df.parquet', 
                  '230301_MIPS_and_DMSO.parquet')
    file_paths = [os.path.join(d, n) for n in file_names]
    df = [pd.read_parquet(p) for p in file_paths]
    df = pd.concat(df).reset_index(drop=True)
    cumulative_density(df)
