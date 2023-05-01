from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy.stats import pearsonr
import seaborn as sns


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
        cols=('rho','theta','phi', 'rho_diff', 'theta_diff', 'phi_diff'), 
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
        sml_df = RobustScaler().fit_transform(sml_df)
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


# ------------------
# tSNE dim reduction
# ------------------

def tsne_embedding(
    df, 
    emb_name, 
    save=None, 
    cols=['rho','theta','phi', 'rho_diff', 'theta_diff', 'phi_diff'],
    umap_cols=None
    ):
    sml_df = df[list(cols)]
    sml_df = sml_df.dropna()
    idxs = sml_df.index.values
    sml_df = sml_df.values
    embedding = TSNE(n_components=2, learning_rate='auto',
                      init='random').fit_transform(sml_df)
    df.loc[idxs, f'tsne_0_{emb_name}'] = embedding[:, 0]
    df.loc[idxs, f'tsne_1_{emb_name}'] = embedding[:, 1]
    if save is not None:
        df.to_parquet(save)
    return df


# --------
# Plotting
# --------

def plot_clusters(
        df, 
        cluster_col, 
        umap_cols, 
        embedding_cols=['rho','theta','phi', 'rho_diff', 'theta_diff', 'phi_diff'], 
        embedding_name='scoords',
        frac=1, 
        frame_range=None, 
        save=None, 
        size=(10, 10), 
        show=True
        ):
    '''
    Plot clusters on a umap
    '''
    # subsample if necessary
    sub_df = df
    if frame_range is not None:
        sub_df = df[(df['frame'] >= frame_range[0]) & (df['frame'] < frame_range[1])]
    if frac < 1:
        sub_df = sub_df.sample(frac=frac)
    labels = pd.unique(sub_df[cluster_col])
    # colour maps for clusters
    try:
        c_map = {
            l : sns.color_palette()[int(l)] for l in labels
        }
    except IndexError:
        col_labs = sub_df[cluster_col].value_counts().index.values[:10]
        other_labs = sub_df[cluster_col].value_counts().index.values[10:]
        cols = [sns.color_palette()[i] for i in range(10)]
        c_map = cmap_dict(col_labs, cols, other_labs, other_col='lightgray')
    # plotting
    fig, ax = plt.subplots()
    ax.scatter(
        sub_df[umap_cols[0]].values, 
        sub_df[umap_cols[1]].values, 
        c=sub_df[cluster_col].map(c_map), s=1)
    #ax.title(f'UMAP projection of platelets clustered according to {embedding_name}', fontsize=18)
    fig.set_size_inches(size[0], size[1])
    # save if necessary
    if save is not None:
        fig.savefig(save)
    # show if required
    if show:
        plt.show()
    return df



def cmap_dict(vals, cols, other_vals, other_col='lightgray'):
    cmap = {vals[i] : cols[i] for i in range(len(vals))}
    for v in other_vals:
        cmap[v] = other_col
    return cmap



def scatter_by_cluster_2D(df, cluster_col, x_col, y_col, frame_range=(190, 193), by_platelet=False, by_frame_exp=True):
    pass
    # want to provide an option for plotting realatonships between variables 
    # good to be able to compar


# -----------------
# PCA Dim Reduction
# -----------------


def PCA_objects(
    df, 
    pca_name,
    cols=('frame', 'rho', 'phi', 'theta', 'rho_diff', 'phi_diff', 'theta_diff')
    ):
    sml_df = df[list(cols)]
    sml_df = sml_df.dropna()
    idxs = sml_df.index.values
    sml_df = StandardScaler().fit_transform(sml_df)
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(sml_df)
    df.loc[idxs, f'PCA_0_{pca_name}'] = pca_results[:, 0]
    df.loc[idxs, f'PCA_1_{pca_name}'] = pca_results[:, 1]
    return df, pca


def PCA_all(
    df, 
    cols
    ):
    sml_df = df[list(cols)]
    sml_df = sml_df.dropna()
    idxs = sml_df.index.values
    sml_df = StandardScaler().fit_transform(sml_df)
    pca = PCA()
    pca_results = pca.fit_transform(sml_df)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()
    return pca


def PCA_corr(
    df, 
    cols, 
    save
    ):
    cols = list(cols)
    sml_df = df[cols]
    sml_df = sml_df.dropna()
    sml_df = StandardScaler().fit_transform(sml_df)
    pca = PCA()
    pca_results = pca.fit_transform(sml_df)
    results = {
        'variables' : cols
    }
    n_iter = len(cols)
    with tqdm(total=n_iter) as progress:
        for comp in range(pca_results.shape[1]):
            nr = f'PC_{comp}_r'
            np = f'PC_{comp}_p'
            rs = []
            ps = []
            for col in cols:
                r, p = pearsonr(df[col].values, pca_results[:, comp])
                rs.append(r)
                ps.append(p)
            results[nr] = rs
            results[np] = ps
            progress.update(1)
    results = pd.DataFrame(results)
    results.to_csv(save)
    plt.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
    plt.show()



def cross_corr(df, cols, save):
    n_iter = len(cols)
    results = {
        'variables' : cols
    }
    heatmap = {
        'variables' : cols
    }
    with tqdm(total=n_iter) as progress:
        for i, col in enumerate(cols):
            n_r = f'{col}_r'
            n_p = f'{col}_p'
            rs = []
            ps = []
            for c in cols:
                r, p = pearsonr(df[col].values, df[c].values)
                rs.append(r)
                ps.append(p)
            results[n_r] = rs
            results[n_p] = ps
            heatmap[col] = rs
            progress.update(1)
    results = pd.DataFrame(results)
    results.to_csv(save)
    heatmap = pd.DataFrame(heatmap)
    heatmap = heatmap.set_index('variables')
    matrix = np.triu(heatmap.values)
    ax = sns.heatmap(heatmap, annot=True,  linewidths=.5, cmap="vlag", vmin=-1, vmax=1, mask=matrix)
    plt.show()
    


# ------------------------------
# Clustering Platelet Properties
# ------------------------------


def cluster_platelets_by_kinetic(
        pdf, # df constructed by construct_platelet_df()
        kin_cols=(
            'end_tort', 
            'sum_cont', 
            'end_path_len', 
           # 'end_track_curv',  
            'nrtracks', 
            'start_frame', 
            'start_rho', 
            'start_phi', 
            'start_theta', 
            'sum_phi', 
            'sum_theta', 
            'sum_nb_cont_15' ), 
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
