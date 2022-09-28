
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import ptitprince as pt
import seaborn as sns

def experiment_VI_plots(
        paths, 
        names, 
        title,
        out_name,
        out_dir,
        cond_ent_over="VI: GT | Output",  
        cond_ent_under="VI: Output | GT", 
        show=True
    ):
    plt.rcParams.update({'font.size': 16})
    groups = []
    ce0 = []
    ce1 = []
    cd = []
    for i, p in enumerate(paths):
        df = pd.read_csv(p)
        ce0.append(df[cond_ent_over].values)
        ce1.append(df[cond_ent_under].values)
        cdpcnt = df['Count difference'].values / df['n_predicted'].values * 100
        cd.append(df['Count difference'].values)
        groups += [names[i]] * len(df)
    x = 'Experiment'
    data = {
        x : groups, 
        cond_ent_over : np.concatenate(ce0), 
        cond_ent_under : np.concatenate(ce1),
        'Count difference' :  np.concatenate(cd),
    }
    data = pd.DataFrame(data)
    o = 'h'
    pal = 'husl'
    sigma = .4
    f, axs = plt.subplots(1, 3, figsize=(12, 6)) #, sharex=True) #, sharey=True)
    ax0 = axs[0]
    ax1 = axs[1]
    ax2 = axs[2]
    pt.RainCloud(x = x, y = cond_ent_over, data = data, palette = pal, bw = sigma,
                 width_viol = 1, ax = ax0, orient = o, width_box=0.4, point_size=5, jitter=1, edgecolor='black')
    ax0.set_title('Over-segmentation conditional entropy')
    pt.RainCloud(x = x, y = cond_ent_under, data = data, palette = pal, bw = sigma,
                 width_viol = 1, ax = ax1, orient = o, width_box=0.4, point_size=5, jitter=1, edgecolor='black')
    ax1.set_title('Under-segmentation conditional entropy')
    pt.RainCloud(x = x, y = 'Count difference', data = data, palette = pal, bw = sigma,
                 width_viol = 1, ax = ax2, orient = o, width_box=0.4, point_size=5, jitter=1, edgecolor='black')
    ax2.set_title('Number of platelets difference')
    f.suptitle(title)
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, out_name + '_VI_rainclould_plots.png')
    plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()


def track_accuracy_raincloudplots(df, x='condition_key', y='pcnt_with_swaps'):
    o = 'h'
    pal = 'husl'
    sigma = .4
    f, ax = plt.subplots(1, 1) #, sharex=True) #, sharey=True)
    ny = 'ID swap frequency (% frames)'
    df = df.rename(columns={y : ny})
    pt.RainCloud(x = x, y = ny, data = df, palette = pal, bw = sigma,
                 width_viol = 1, ax = ax, orient = o, width_box=0.4, point_size=5, jitter=1, edgecolor='black')
    ax.set_title('Percentage of frames with ID swaps')
    plt.show()


def track_accuracy_barchart(df, x='condition_key', y='pcnt_with_swaps'):
    f, ax = plt.subplots(1, 1)
    ny = 'ID swap frequency (% frames)'
    df = df.rename(columns={y : ny})
    sns.barplot(x=x, y=ny, data=df, ax=ax, capsize=.2, palette='husl')
    plt.show()


if __name__ == '__main__':
    cang_d = '/Users/amcg0011/Data/platelet-analysis/DL-validation/cang'
    dmso_d = '/Users/amcg0011/Data/platelet-analysis/DL-validation/DMSO'
    saline_d = '/Users/amcg0011/Data/platelet-analysis/DL-validation/saline'

    cang_p = os.path.join(cang_d, 'cang_segmentation-metrics_metrics.csv')
    dmso_p = os.path.join(dmso_d, 'DMSO_segmentation-metrics_metrics.csv')
    saline_p = os.path.join(saline_d, 'saline-segmentation-metrics_metrics.csv')

    cang = pd.read_csv(cang_p)
    dmso = pd.read_csv(dmso_p)
    saline = pd.read_csv(saline_p)

    paths = [saline_p, cang_p, dmso_p]
    names = ['saline', 'cangrelor', 'dmso']
    title = 'VI for saline, cangrelor, and DMSO segmentations'
    out_name = 'VI_saline-cang-dmso_autogen.svg'
    out_dir = '/Users/amcg0011/Data/platelet-analysis/DL-validation'

    #experiment_VI_plots(
     #   paths, 
       # names, 
      #  title,
       # out_name,
       # out_dir,
       # show=True
    #)

    def print_mean_and_sem(df):
        print('under-segmentation')
        print(df['VI: GT | Output'].mean())
        print(df['VI: GT | Output'].sem())
        print('over-segmentation')
        print(df['VI: Output | GT'].mean())
        print(df['VI: Output | GT'].sem())

    #print('saline')
    #print_mean_and_sem(saline)
    #print('cang')
    #print_mean_and_sem(cang)
    #print('dmso')
    #print_mean_and_sem(dmso)

    p = '/Users/amcg0011/Data/platelet-analysis/track-accuracy/saline_dmso_cang.csv'
    df = pd.read_csv(p)
    #track_accuracy_raincloudplots(df)
    #track_accuracy_barchart(df)

    def print_mean_and_sem_1(df):
        for k, g in df.groupby(['condition_key', ]):
            print(k)
            print('mean: ', g['pcnt_with_swaps'].mean())
            print('sem: ', g['pcnt_with_swaps'].sem())
            print('n f terms: ', g['false_term'].sum())
    print_mean_and_sem_1(df)


def plot_PH(data, r):
    f, ax = plt.subplots(1, 1)
    ax.set_ylim((-13, 13))
    ax.set_xlim((-13, 13))
    ax.scatter(data[:, 0], data[:, 1], s = r)
    ax.scatter(data[:, 0], data[:, 1], color = 'black')
    plt.show()

