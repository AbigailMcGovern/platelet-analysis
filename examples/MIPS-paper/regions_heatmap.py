from plateletanalysis.analysis.summary_data import regions_heatmap_data
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# ------------------
# Get data from file
# ------------------
epochs = False
if epochs:
    save_path = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/summary_data/230612_exp_region_epoch_data.csv'
if not epochs:
    save_path = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/summary_data/230612_exp_region_phase_data.csv'

# change variables
result = pd.read_csv(save_path)
result['average platelet sliding (um)'] = result['average platelet sliding (um)'] / 0.32
result['initial platelet velocity change (um/s)'] = - result['initial platelet velocity change (um/s)']
result['total change in velocity (um/s)'] = - result['total change in velocity (um/s)']

# variables and variable order
vars_dict = {
    'platelet count' : 'count', 
    'platelet average density (um^-3)' : 'density', 
    'platelet density gain (um^-3)' : 'density gain', 
    'average platelet instability' : 'instability', 
    'average net platelet loss (/min)' : 'net platelet loss',
    'average platelet tracking time (s)' : 'time tracked',
    'P(< 15s)' : 'P(< 15s)',
    'P(> 60s)' : 'P(> 60s)', 
    'recruitment' : 'recruitment', 
    'shedding' : 'shedding', 
    'P(recruited < 15 s)' : 'P(recruited < 15 s)', 
    'P(recruited > 60 s)': 'P(recruited > 60 s)', 
    'average platelet y velocity (um/s)' : 'y-axis velocity', 
    'total change in velocity (um/s)' : 'net decceleration', 
    'initial platlet density (um^-3)' : 'initial density', 
    'initial platelet instability' : 'initial instability',
    'initial platelet velocity change (um/s)' : 'initial decelleration', 
}

# heatmap data (consolidation)
if not epochs:
    result = result[result['phase'] == 'consolidation']
    save_path = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/summary_data/230614_regions_heatmap_data_MIPS_ord.csv'
    heatmap_df = regions_heatmap_data(result, save_path, vars_dict, group='MIPS')

# heatmap data (epochs)
if epochs:
    heatmap_df = []
    epochs = pd.unique(result['epoch'])
    for e in epochs:
        res = result[result['epoch'] == e]
        save_path = f'/Users/abigailmcgovern/Data/platelet-analysis/MIPS/summary_data/230614_regions_heatmap_data_MIPS_ord_{e}.csv'
        sdf = regions_heatmap_data(res, save_path, vars_dict, group='MIPS')
        sdf['region x rank'] = sdf['region x rank'] + ': ' + e
        #sdf = sdf.drop(columns='epoch')
        heatmap_df.append(sdf)
    heatmap_df = pd.concat(heatmap_df).reset_index(drop=True)

# plotting
def regions_heatmap(df):
    fig, ax = plt.subplots(1, 1)
    df = df.set_index('region x rank')
    #columns = df.columns.values
    #for col in columns:
       # df[col] = df[col].apply(_fold_change)
    sns.heatmap(data=df, annot=False, center=100, cmap='bwr', ax=ax, vmax=200)
    fig.set_size_inches(8, 8)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.97, bottom=0.2)
    plt.show()

def _fold_change(val):
    v = (val - 100) / 100
    if v < 0:
        v = 1 / v
    return v

regions_heatmap(heatmap_df)