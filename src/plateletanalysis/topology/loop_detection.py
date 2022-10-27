from ripser import ripser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import seaborn as sns
from pathlib import Path
from scipy import stats
from plateletanalysis.variables.measure import quantile_normalise_variables_frame



# ------------------
# Number of features
# ------------------

def number_outliying_over_time(df, save_path, centile=75, col='nb_density_15_pcntf', y_col='ys_pcnt', x_col='x_s_pcnt', units='%'):
    out = get_outlier_number(df, x_col, y_col, units)
    out = pd.DataFrame(out)
    out.to_csv(save_path)
    x = 'time (s)'
    y0 = f'upper bound {units}'
    y1 = 'n outliers'
    out['measure'] = ['n outliers', ] * len(out)
    y2 = 'max outlier / mean outlier'
    sns.set_style("ticks")
    fig, axes = plt.subplots(3, 1, sharex=True)
    ax0, ax1, ax2 = axes.ravel()
    e0 = sns.lineplot(x, y0, data=out, ax=ax0)
    e1 = sns.lineplot(x, y1, data=out, ax=ax1, hue='measure')
    e2 = sns.lineplot(x, y2, data=out, ax=ax2)
    plt.show()


def get_outlier_number(data, x_col, y_col, units):
    frames = np.arange(data['frame'].max())
    paths = pd.unique(data['path'])
    n_outliers = []
    outlier_upper = []
    outlier_max_div_mean = []
    time = frames / 0.321764322705706
    time = np.concatenate([time.copy() for _ in range(len(paths))])
    with tqdm(total=len(paths) * len(frames)) as progress:
        for p in paths:
            pdf = data[data['path'] == p]
            for t in frames:
                data_t = pdf[pdf['frame'] == t]
                X = data_t[[x_col, y_col]].values
                if len(X) > 0:
                    dgms = ripser(X)['dgms']
                    h1 = dgms[1]
                    #print(h1)
                    if len(h1) > 0:
                        diff = h1[:, 1] - h1[:, 0]
                        #IQR = iqr(diff)
                        #Q3 = scoreatpercentile(diff, 75)
                        #upper = Q3 + (1.5 * IQR)
                        mean = np.mean(diff)
                        std = np.std(diff)
                        upper  = mean + (std * 5)
                        idxs = np.where(diff > upper)[0]
                        n = len(idxs)
                        outliers = diff[idxs]
                        if n > 0:
                            i = np.argmax(outliers)
                            max = outliers[i]
                            outliers = np.delete(outliers, i)
                            if len(outliers) > 0:
                                o_mean = np.mean(outliers)
                                max_div_mean = max / o_mean # expect close to 1 on average in toroidal phase
                            else:
                                max_div_mean = 1
                        else: 
                            max_div_mean = 0 # really this is undefined, just want to know if the top outlier is very far from others
                        n_outliers.append(n)
                        outlier_upper.append(upper)
                        outlier_max_div_mean.append(max_div_mean)
                    else:
                        n_outliers.append(np.NaN)
                        outlier_upper.append(np.NaN)
                        outlier_max_div_mean.append(np.NaN)
                else:
                    n_outliers.append(np.NaN)
                    outlier_upper.append(np.NaN)
                    outlier_max_div_mean.append(np.NaN)
                progress.update(1)
    x = 'time (s)'
    out = {
        x : time, 
        f'upper bound {units}' : outlier_upper, 
        'n outliers' : n_outliers, 
        'max outlier / mean outlier' : outlier_max_div_mean, 
    }
    return out