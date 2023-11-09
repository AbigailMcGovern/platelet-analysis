import pandas as pd
import numpy as np



def get_max_times(df, sp, txs=('DMSO (MIPS)', 'DMSO (SQ)', 'saline')):
    all_df = df.sort_values('time (s)')
    out = {
        'treatment' : [], 
        'max_mean_count' : [], 
        'time_max_count' : []
    }
    for tx in txs:
        df = all_df[all_df['treatment'] == tx]
        data = {
            'time (s)' : [], 
            'platelet count raw' : []
        }
        for t, grp in df.groupby('time (s)'):
            data['time (s)'].append(t)
            data['platelet count raw'].append(grp['platelet count raw'].mean())
        data = pd.DataFrame(data)
        roll = data['platelet count raw'].rolling(8, center=False).mean()
        idx = np.argmax(roll)
        max_count = roll[idx]
        max_time = data['time (s)'][idx]
        out['treatment'].append(tx)
        out['max_mean_count'].append(max_count)
        out['time_max_count'].append(max_time)
        print(f'{tx} max count and time ')
        print(max_time, 's: ', max_count)
    out = pd.DataFrame(out)
    out.to_csv(sp)
    return out


p = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/Figure_3/230420_count-and-growth-pcnt_rolling-counts.csv'
df = pd.read_csv(p)
sp = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/summary_data/230612_max_count_and_time.csv'
out = get_max_times(df, sp, txs=('DMSO (MIPS)', 'DMSO (SQ)', 'saline'))
