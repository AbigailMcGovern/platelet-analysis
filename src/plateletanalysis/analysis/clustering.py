from cProfile import label
from enum import unique
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import random

# ---------------------------
# Clustering Kinetic Profiles
# ---------------------------

def cluster_spherical_coordinates(df, save_checkpoint=None):
    '''
    DBSCAN Clustering of platelets in a single data frame based on 
    shpherical coordinates and spherical velocities. 

    Parameters
    ----------
    '''
    min_samples = 5
    #eps_list = [5,7.5,10,15,20] # will prob have to be adjusted because phi and theta are radians (maybe could adjust to degrees??)
    eps_list = [.1, 0.5, .75, 1]
    for eps in eps_list:
        vals = [0, ] * len(df)
        if save_checkpoint is None:
            df[f'dbscan_{eps}'] = np.array(vals, dtype=int)
        else:
            if f'dbscan_{eps}' not in df.columns.values:
                df[f'dbscan_{eps}'] = np.array(vals, dtype=int)
    cols = ['rho','theta','phi', 'rho_diff', 'theta_diff', 'phi_diff']
    # nah actually, Ima standardise the data to a z score
    # can't hurt ...
    try:
        df = df.set_index('pid')
    except KeyError:
        df['pid'] = range(len(df))
        df = df.set_index('pid')
    sml_df = df[cols]
    sml_df = sml_df.dropna()
    idxs = sml_df.index.values
    data = StandardScaler().fit_transform(sml_df)
    #data = sml_df[cols].values 
    del sml_df
    with tqdm(total=len(eps_list)) as progress:
        for eps in eps_list:
            #print(df[f'dbscan_{eps}'].max(), df[f'dbscan_{eps}'].min())
            do_eps = df[f'dbscan_{eps}'].max() == 0 and df[f'dbscan_{eps}'].min() == 0
            if do_eps:
                df = get_DBSCAN_clusters(data, eps, min_samples, df, idxs, f'dbscan_{eps}')
                if save_checkpoint is not None:
                    df.to_parquet(save_checkpoint)
            progress.update(1)
    df = df.reset_index()
    return df


def umap_dbscan_cluster(
        df, 
        emb_name, 
        save=None, 
        eps_list=(0.1, .25, 0.5, 0.75, 1), 
        min_samples=5,
        cols=('frame', 'rho','theta','phi', 'rho_diff', 'theta_diff', 'phi_diff'), 
        frame_range=None, 
        use_checkpoint=False,
        umap_cols=None
        ):
    cluster_cols = [f'umap_0_{emb_name}', f'umap_1_{emb_name}']
    df, data, idxs = umap_embedding(df, emb_name, save, cols, umap_cols)
    df, eps_cols = write_eps_if_need_to(df, eps_list, emb_name, use_checkpoint)
    df = DBSCAN_for_eps_list(data, idxs, df, eps_list, eps_cols, min_samples, use_checkpoint, save)
    return df




def write_eps_if_need_to(df, eps_list, emb_name, use_checkpoint):
    vals = [0, ] * len(df)
    col_names = []
    for eps in eps_list:
        col_name = f'{emb_name}_udbscan_{eps}'
        col_names.append(col_name)
        if col_name not in df.columns.values:
            df[col_name] = np.array(vals, dtype=int)
        else:
            if not use_checkpoint: # overwrite values if we arent using a checkpoint
                df[col_name] = np.array(vals, dtype=int)
    return df, col_names



def DBSCAN_for_eps_list(data, idxs, df, eps_list, eps_cols, min_samples, use_checkpoint, save):
     with tqdm(total=len(eps_list)) as progress:
        for i, eps in enumerate(eps_list):
            #print(df[f'dbscan_{eps}'].max(), df[f'dbscan_{eps}'].min())
            do_eps = df[eps_cols[i]].max() == 0 and df[eps_cols[i]].min() == 0
            if not use_checkpoint:
                do_eps = True
            if do_eps:
                df = get_DBSCAN_clusters(data, eps, min_samples, df, idxs, eps_cols[i])
                if save is not None:
                    df.to_parquet(save)
            progress.update(1)
        return df


def get_DBSCAN_clusters(data, eps, min_samples, df, idxs, eps_col):
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=6).fit(data)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    info = f'eps {eps}: found {n_clusters_} clusters with {n_noise_} / {data.shape[0]} points unassigned'
    print(info)
    labels = labels.astype(int) + 1
    print(np.unique(labels))
    df.loc[idxs, eps_col] = labels.astype(int)
    return df



def cluster_platelets_kinetic(df):
    max_frame = df['frame'].max() + 1
    coord_cols = ['rho','theta','phi', 'rho_diff', 'theta_diff', 'phi_diff']
    df_gb = df.groupby(['path', 'particle'])
    n_platelets = len(df_gb.mean())
    platelet_ID = range(n_platelets)
    p_df = {'platelet_ID' : platelet_ID}
    for col in coord_cols:
        for t in range(max_frame):
            if 'diff' in col:
                p_df[f'{col}_t{t}'] = np.array([0, ] * n_platelets, dtype=np.float64)
            else: # coordinates without 
                p_df[f'{col}_t{t}'] = np.array([-1, ] * n_platelets, dtype=np.float64)
    idxs = pd.unique(df.index)
    p_df = pd.DataFrame(p_df)
    for i in idxs: # i ~ (path, particle)
        lil_df = df.loc[i]
        frames = lil_df['frame'].values
        for col in coord_cols:
            for t in frames:
                pass

    
# ------------------
# UMAP Dim Reduction
# ------------------


def umap_embedding(
        df, 
        emb_name, 
        save=None, 
        cols=['rho','theta','phi', 'rho_diff', 'theta_diff', 'phi_diff'],
        umap_cols=None
    ):
    sml_df = df[list(cols)]
    sml_df = sml_df.dropna()
    idxs = sml_df.index.values
    if umap_cols is None:
        # standardise the data to a z score
        sml_df = StandardScaler().fit_transform(sml_df)
        # going to reduce the data using umap
        reducer = umap.UMAP()
        embedding = reducer.fit(sml_df)
        embedding = embedding.embedding_
        df.loc[idxs, f'umap_0_{emb_name}'] = embedding[:, 0]
        df.loc[idxs, f'umap_1_{emb_name}'] = embedding[:, 1]
    else:
        emb_df = df[umap_cols].loc[idxs]
        embedding = emb_df[umap_cols].values
    if save is not None:
        df.to_parquet(save)
    return df, embedding, idxs


# --------
# Plotting
# --------

def plot_clusters(
        df, 
        cluster_col, 
        umap_cols=None, 
        embedding_cols=['rho','theta','phi', 'rho_diff', 'theta_diff', 'phi_diff'], 
        embedding_name='scoords',
        frac=1, 
        frame_range=(0, 194)):
    '''
    Plot clusters on a umap
    '''
    if umap_cols is None:
        try:
            df = df.set_index('pid')
        except KeyError:
            df['pid'] = range(len(df))
            df = df.set_index('pid')
        cols = embedding_cols
        sml_df = df[cols]
        sml_df = sml_df.dropna()
        idxs = sml_df.index.values
        # standardise the data to a z score
        sml_df = StandardScaler().fit_transform(sml_df)
        # going to reduce the data using umap
        reducer = umap.UMAP()
        embedding = reducer.fit(sml_df)
        embedding = embedding.embedding_
        df.loc[idxs, f'umap_0_{embedding_name}'] = embedding[:, 0]
        df.loc[idxs, f'umap_1_{embedding_name}'] = embedding[:, 1]
        umap_cols = [f'umap_0_{embedding_name}', f'umap_1_{embedding_name}']
        del sml_df
    sub_df = df[(df['frame'] >= frame_range[0]) & (df['frame'] < frame_range[1])]
    if frac < 1:
        sub_df = sub_df.sample(frac=frac)
    labels = sub_df[cluster_col].values
    try:
        c_map = [sns.color_palette()[int(label)] for label in labels]
    except IndexError:
        import matplotlib.colors as mcolours
        colours = [n for n, c in mcolours.CSS4_COLORS.items()]
        colours = [col for col in colours if 'white' not in col and 'gray' not in col]
        random.shuffle(colours)
        c_map = [colours[int(label)] for label in labels]
    plt.scatter(
        sub_df[umap_cols[0]].values, 
        sub_df[umap_cols[1]].values, 
        c=c_map, s=1)
    plt.gca().set_aspect('equal', 'datalim')
    plt.title(f'UMAP projection of platelets clustered according to {embedding_name}', fontsize=18)
    plt.show()
    df.reset_index()
    return df

    # im not totally sure how to display this. 
    # I can use a umap to reduce the data to two dimensions to give the viewer a visual intuition
    # I think it might also be useful to plot the clusters on 2D cross sections

    # may need to take a random sample of points to reduce the total number of data points (there are too many to display confortably)
    # could display points across a number of frames or at a single time frame
    # random sub-sampling probably necessary 



def scatter_by_cluster_2D(df, cluster_col, x_col, y_col, frame_range=(190, 193), by_platelet=False, by_frame_exp=True):
    pass
    # want to provide an option for plotting realatonships between variables 
    # good to be able to compar


# ------------------------------
# Clustering Platelet Properties
# ------------------------------


def cluster_platelets_by_kinetic(
        pdf, # df constructed by construct_platelet_df()
        kin_cols=(
            'end_tort', 
            'c_cont', 
            'end_track_len', 
            'end_track_curv',  
            'nrtracks', 
            'start_frame', 
            'start_rho', 
            'start_phi', 
            'start_theta', 
            'c_phi', 
            'c_theta', 
            'c_nb_cont_15' ), 
        emb_name='kin-0'
    ):
    df = umap_dbscan_cluster(pdf, emb_name, 
        save=None, 
        eps_list=(0.1, .25, 0.5, 0.75, 1), 
        min_samples=5,
        cols=('frame', 'rho','theta','phi', 'rho_diff', 'theta_diff', 'phi_diff'), 
        frame_range=None, 
        use_checkpoint=False,
        umap_cols=None
        )



# -------------------------
# Helper variable functions
# -------------------------

def construct_platelet_df(df, cols, save=None):
    '''
    Highly inefficient function to generate platelet DF. Will make more efficient only if
    too time intensive when run. 
    '''
    df = df[df['tracked'] == True] # only interested in tracked platelets
    df_gb = df.groupby(['path', 'particle'])
    idx = pd.unique(df_gb.index.values)
    plate_id = range(len(idx))
    pdf = {
        'plate_id' : plate_id, 
        'path' : [i[0] for i in idx], 
        'particle' : [i[1] for i in idx], 
        'gbindex' : idx
        }
    pdf = pd.DataFrame(pdf)
    pdf.set_index('gbindex')
    for col in cols:
        if col.startswith('c_'):
            if col not in df.columns.values:
                bcol = col[2:]
                df = cumulative_platelet_score(df, bcol)
                df_gb = df.groupby(['path', 'particle'])
            sml_df = df_gb[df_gb['terminating'] == True] # each platelet should only have one terminating track
            c_idxs = sml_df.index.values
            pdf.loc[c_idxs, col] = sml_df[col].values
        elif col.startswith('start_'):
            bcol = col[6:]
            s_idxs, vals = start_track_value(df, bcol)
            pdf.loc[s_idxs, col] = vals
        elif col.startswith('end_'):
            bcol = col[4:]
            s_idxs, vals = end_track_value(df, bcol)
            pdf.loc[s_idxs, col] = vals
        elif col.startswith('var_'):
            pass # get varience (need to implement method)
        else:
            means = df_gb.mean()
            m_idxs = means.index.values
            vals = means[col].values
            pdf.loc[m_idxs, col] = vals



def cumulative_platelet_score(df, col):
    df_gb = df.groupby(['path', 'particle'])
    idx = pd.unique(df_gb.index.values)
    df.set_index('pid')
    for i in idx:
        sml_df = df_gb[i]
        sml_df = sml_df.sort_values('frame')
        result = sml_df[col].cumsum()
        pids = sml_df['pid'].values
        df[pids, f'c_{col}']
    return df


def start_track_value(df, col):
    df_gb = df.groupby(['path', 'particle'])
    df_gb = df_gb[df_gb['tracknr'] == 1] # each platelet should only have one start track
    idx = pd.unique(df_gb.index.values)
    vals = df[col].values
    return idx, vals


def end_track_value(df, col):
    df_gb = df.groupby(['path', 'particle'])
    df_gb = df_gb[df_gb['terminating'] == True] # each platelet should only have one terminating track
    idx = pd.unique(df_gb.index.values)
    vals = df[col].values
    return idx, vals


# ----------
# Old DBSCAN
# ----------

def DBSCAN_cluster_1(pc):
    min_samples=5
    eps_list=[5,7.5,10,15,20]
    cl_=[]
    X=pc[['x_s','ys','zs']].values # 3D
    for eps in eps_list:
        # Compute DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        cl_.append(pd.DataFrame({('cl_idx_' + str(eps) ) : (labels)}))
    cl_.append(pc.reset_index().pid)
    return (pd.concat(cl_, axis=1))


def DBSCAN_cluster_2(pc):
    min_samples=5
    eps_list=np.arange(3, 31, 2)
    cl_=[]
    X=pc[['x_s','ys','zs']].values # 3D
    for eps in eps_list:
        # Compute DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        labels[labels>-1]=eps
        labels[labels==-1]=42
        cl_.append(pd.DataFrame({('cl_idx_' + str(eps) ) : (labels)}))
    cld_grp=pd.concat(cl_, axis=1)
    cld_grp=pd.DataFrame({('cld' ) : (cld_grp.min(axis=1))})
    cld_grp['pid']=(pc.reset_index().pid)
    return (cld_grp)
