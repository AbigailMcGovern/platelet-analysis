import numpy as np
from scipy.signal import find_peaks, peak_widths, peak_prominences
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.fft import fft, fftfreq



def rolling_variable(df, p='path', t='time (s)', y='corrected calcium', window=20):
    n = y + ' rolling'
    df = df.copy()
    for k, g in df.groupby([p]):
        g = g.sort_values(t)
        idx = g.index.values
        rolling = g[y].rolling(window=window,win_type='triangular',min_periods=1,center=False).mean()
        df.loc[idx, n] = rolling
    return df



def plot_experiment_calcium(df, ca_col='corrected calcium', treatments=('saline', 'bivalirudin', 'cangrelor'), windows=(15, 20, 25, 30)):
    #df = pd.concat([df[df['treatment'] == tx] for tx in treatments])
    fig, axs = plt.subplots(len(treatments), len(windows), sharex=True, sharey=True)
    for i in range(len(treatments)):
        for j in range(len(windows)):
            tx = treatments[i]
            w = windows[j]
            ax = axs[i, j]
            sml_df = df[df['treatment'] == tx]
            if w > 2:
                sml_df = rolling_variable(sml_df, window=w, y=ca_col)
                y = f'{ca_col} rolling'
            else:
                y = ca_col
            sns.lineplot(x = 'time (s)', y = y, hue='path', data=sml_df, ax=ax, legend=False)
    plt.show()



def cumulative_calcium(df, ca_col='corrected calcium', t='time (s)'):
    n = ca_col + ' csum'
    df = df.copy()
    for k, g in df.groupby(['path']):
        g = g.sort_values(t)
        idx = g.index.values
        cumsum = g[ca_col].cumsum()
        df.loc[idx, n] = cumsum
    return df
        


def find_peak_info(df, save_path, plot=True):
    df = rolling_variable(df)
    ca_roll = {k : sml_df['corrected calcium rolling'].values for k, sml_df in df.groupby(['path'])}
    peak_idx = {k : find_peaks(ca_roll[k])[0] for k in ca_roll.keys()}
    time = {k : [sml_df['time (s)'].values[i] for i in peak_idx[k]] for k, sml_df in df.groupby(['path'])}
    tx = {k : [sml_df['treatment'].values[i] for i in peak_idx[k]] for k, sml_df in df.groupby(['path'])}
    peak_vals = {}
    for k in ca_roll.keys():
        idxs = peak_idx[k]
        ca = ca_roll[k]
        vals = [ca[i] for i in idxs]
        peak_vals[k] = vals
    peak_w = {k : peak_widths(ca_roll[k], peak_idx[k]) for k in ca_roll.keys()}
    # returns (width, width_height [[left_ips, right_ips], ...])
    peak_prom = {k : peak_prominences(ca_roll[k], peak_idx[k]) for k in ca_roll.keys()}
    # returns (prominance, [[left_base, right_base], ...])
    peak_info = {
        'path' : [], 
        'treatment' : [], 
        'time (s)' : [],
        'index' : [], 
        'value' : [], 
        'width' : [], 
        'width_height' : [], 
        'prominence' : [], 
        'left_base' : [], 
        'right_base' : [], 
    }
    for k in ca_roll.keys():
        for i in range(len(peak_idx[k])):
            peak_info['path'].append(k)
            peak_info['treatment'].append(tx[k][i])
            peak_info['time (s)'].append(time[k][i])
            peak_info['index'].append(peak_idx[k][i])
            peak_info['value'].append(peak_vals[k][i])
            peak_info['width'].append(peak_w[k][0][i])
            peak_info['width_height'].append(peak_w[k][1][i])
            peak_info['prominence'].append(peak_prom[k][0][i])
            peak_info['left_base'].append(ca_roll[k][peak_prom[k][1][i]])
            peak_info['right_base'].append(ca_roll[k][peak_prom[k][2][i]])
    peak_info = pd.DataFrame(peak_info)
    peak_info.to_csv(save_path)
    txs = pd.unique(peak_info['treatment'])
    if plot:
        fig, axs = plt.subplots(len(txs), 1, sharey=True, sharex=True)
        for i, ax in enumerate(axs):
            sml_df = df[df['treatment'] == txs[i]]
            tpaths = pd.unique(sml_df['path'])
            x = sml_df['time (s)'].values
            y = sml_df['corrected calcium rolling']
            for p in tpaths:
                pdf = sml_df[sml_df['path'] == p]
                x = pdf['time (s)'].values
                y = pdf['corrected calcium rolling']
                ax.plot(x, y)
                peak_df = peak_info[peak_info['treatment'] == txs[i]]
                px = peak_df['time (s)'].values
                py = peak_df['value'].values
                ax.scatter(px, py, s=4)
                ax.set_xlabel(txs[i])
        plt.show()
    peak_info.to_csv(save_path)
    return peak_info



def find_wave_descriptors(peaks, df):
    out = {
        'path' : [], 
        'n_peaks' : [], 
        'av_width' : [], 
        'av_value' : [], 
        'av_left_base' : [], 
        'av_right_base' : [], 
        'av_prominence' : [], 
        'delay_to_max' : [], 
        'delay_to_min' : [], 
    }
    for p, grp in peaks.groupby(['path']):
        # n peaks
        pass
        # average value
        # average width
        # average left base
        # average right base
        # average prominence
        # delay to max peak
        # delay to min peak
        # delay to highest promnience peak
        # delay to max width peak
        #
    pass



def persistence_analysis(df):

    pass



def treatment_wise_spectral(df):
    txs = pd.unique(df['treatment'])
    out = {
        'treatment' : [], 
        'frequency' : [], 
        'magnetude' : [], 
    }
    for tx in txs:
        txdf = df[df['treatment'] == tx]
        paths = pd.unique(txdf['path'])
        res = spectral_analysis(df, paths, plot=False)
        out['treatment'] = out['treatment'] + [tx, ] * len(res['frequency'])
        out['frequency'] = np.concatenate([out['frequency'], res['frequency']])
        out['magnetude'] = np.concatenate([out['magnetude'], res['magnetude']])
    sns.lineplot(x='frequency', y='magnetude', hue='treatment', data=out)
    plt.show()



def spectral_analysis(df, paths, ca_col='corrected calcium', plot=True):
    out = {
        'frequency' : [], 
        'magnetude' : [], 
    }
    for p in paths:
        df0 = df[df['path'] == p]
        N = len(df0)
        df0 = df0.sort_values('time (s)')
        y = df0[ca_col].values
        yf = fft(y)
        x = df0['time (s)'].values
        T = x[1] - x[0]
        xf = fftfreq(N, T)[:N//2]
        if plot:
            plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
        yf = 2.0/N * np.abs(yf[0:N//2])
        out['magnetude'] = np.concatenate([out['magnetude'], yf])
        out['frequency'] = np.concatenate([out['frequency'], xf])
    if plot:
        plt.show()
    return out




if __name__ == '__main__':
    import os
    d = '/Users/amcg0011/Data/platelet-analysis/TDA/treatment_comparison'
    n = 'saline_biva_cang_sq_mips_PH-data-all.csv'
    p = os.path.join(d, n)
    df = pd.read_csv(p)
    #df = cumulative_calcium(df)
    plot_experiment_calcium(df, treatments=('saline', 'bivalirudin', 'cangrelor'), ca_col='corrected calcium', windows=(0, 5, 15, 20))
    #save_path = '/Users/amcg0011/Data/platelet-analysis/calcium_waves/peak-data.csv'
    #find_peak_info(df, save_path, plot=True)