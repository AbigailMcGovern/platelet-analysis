import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


# import matplotlib.pyplot as plt
# df_gb = df.groupby(['path', 'frame']).mean()
# frames = [i[1] for i in df_gb.index.values]
# paths = [i[0] for i in df_gb.index.values]
# upaths = list(set(paths))
# import matplotlib.colors as mcolours
# colours = [n for n, c in mcolours.CSS4_COLORS.items()]
# colours_n15 = ['darkgreen', 'mediumseagreen', 'darkslateblue', 'cadetblue', 'darkcyan', 'darkorchid', 'springgreen', 'mediumturquoise', 'palevioletred', 'midnightblue', 'crimson', 'orange', 'sienna', 'firebrick', 'gold']
# map_col = {p : c for p, c in zip(upaths, colours_n15)}

# df_gb['path'] = paths
# df_gb['frame'] = frames

# plot_scatter_3d(sub_df, 'umap_0_SCnV', 'umap_1_SCnV', 'frame', map_col, 'UMAP 0', 'UMAP 1', 'frame')

def plot_scatter(df, x_col, y_col, c_map, title, xlab, ylab, c_col='path'):
    plt.scatter(df[x_col], df[y_col], c=df[c_col].map(c_map), s=1)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()
    
def plot_scatter_map(df, x_col, y_col, c_map, c_col):
    plt.scatter(df[x_col], df[y_col], c=df[c_col].map(c_map), s=1)
    plt.title(c_col)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()


def plot_scatter_3d(df, x_col, y_col, z_col, c_map, xlab, ylab, zlab, c_col='path', map=True):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    if map:
        scatter = ax.scatter3D(df[x_col], df[y_col], df[z_col], c=df[c_col].map(c_map), s=1)
    else:
        scatter = ax.scatter3D(df[x_col], df[y_col], df[z_col], c=c_map, s=1)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_zlabel(zlab)
    plt.show()


def plot_scatter_3d_1(df, x_val, y_val, z_val, c_map, xlab, ylab, zlab, c_col='path'):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    scatter = ax.scatter3D(x_val, y_val, z_val, c=df[c_col].map(c_map))
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_zlabel(zlab)
    plt.show()


def exp_by_frame_timerange_scatter(
        df, 
        x_col, 
        y_col, 
        min_frame, 
        max_frame, 
        c_map, 
        x_lab, 
        y_lab,
        x_group='mean', 
        y_group='mean', 
        c_col='path'
    ):
    fs_df = []
    frames = []
    for frame in range(min_frame, max_frame):
        f_df = df[df['frame'] == frame]
        fs_df.append(f_df)
    fs_df = pd.concat(fs_df)
    if x_group == 'mean':
        x_df = fs_df.groupby(['path', 'frame']).mean()
    elif x_group == 'sem':
        x_df = fs_df.groupby(['path', 'frame']).sem()
    x_vals = x_df[x_col].values
    paths = [i[0] for i in x_df.index.values]
    x_df['path'] = paths
    frames = [i[1] for i in x_df.index.values]
    x_df['frame'] = frames
    if y_group == 'mean':
        y_vals = fs_df.groupby(['path', 'frame']).mean()[y_col].values
    elif y_group == 'sem':
        y_vals = fs_df.groupby(['path', 'frame']).sem()[y_col].values
    plt.scatter(x_vals, y_vals, c=x_df[c_col].map(c_map))
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.show()



def basic_scatter(df, x_col, y_col, c_col, v_min=None, v_max=None):
    if v_min is None or v_max is None:
        plt.scatter(df[x_col], df[y_col], s=1, c=df[c_col])
    else:
        plt.scatter(df[x_col], df[y_col], s=1, c=df[c_col], vmin=v_min, vmax=v_max)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.colorbar()
    plt.title(c_col)
    plt.show()

    # scatter @ over a select number of frames

# plot_scatter(df_190_ii, 'ca_corr_diff', 'nb_density_15', map_col, 'Platelets at f190 - calcium derivative versus local density derivative', 'Ca2+ derivative', 'local density')

# exp_by_frame_timerange_scatter(
#        df, 
#        'ca_corr_diff', 
#        'nb_density_15', 
#        100, 
#        193, 
#        map_col, 
#        'Ca2+ derivative SEM', 
#        'local density mean',
#        x_group='sem', 
#        y_group='mean', 
#        c_col='path'
#    )

#plot_scatter_3d(df_gb, 'frame', 'nb_density_15', 'ca_corr', map_col, 'Frame', 'Local density', 'Corrected calcium', c_col='path')
#plot_scatter(df_gb, 'ca_corr', 'nb_density_15', map_col, 
# 'Experiment-by-frame mean corrected calcium versus local density', 
# 'Corrected calcium fluorescence', 'Local density (/um**3)')
# plot_scatter(df_gb, 'ca_corr_diff', 'nb_density_15_diff', map_col, 'Experiment-by-frame mean calcium derivative versus local density derivative', 'Mean Ca2+ derivative', 'Mean local density derivative')

# plot_scatter_3d(df_gb, 'frame', 'nb_density_15_diff', 'ca_corr_diff', map_col, 'Frame', 'Local density diff', 'Corrected calcium diff')

#plot_scatter_3d_1(df_gb, df_gb['frame'], df_gb['nb_density_15'], df_gb_sem['ca_corr'], map_col, 'Frame', 'Local density mean', 'Corrected calcium SEM', c_col='path')


def cmap_dict(vals, cols, other_vals, other_col='lightgray'):
    cmap = {vals[i] : cols[i] for i in range(len(vals))}
    for v in other_vals:
        cmap[v] = other_col
    return cmap

def recode_dict(vals, other_vals, other_code=0):
    codes = {vals[i] : i for i in range(len(vals))}
    for v in other_vals:
        codes[v] = other_code
    return codes

    #codes = demo_df['SCnV_add_0_udbscan_0.1'].map(

def path_value_counts(df):
    paths = pd.unique(df['path'])
    for path in paths:
        pdf = df[df['path'] == path]
        print(path)
        print(pdf['SCnV_add_1_udbscan_0.1'].value_counts())