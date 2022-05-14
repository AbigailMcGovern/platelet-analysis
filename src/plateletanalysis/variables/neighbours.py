from asyncio import as_completed
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from toolz import curry
from tqdm import tqdm
from distributed import Client, as_completed


# ------------------
# Finding Neighbours
# ------------------
# Feat. cKD tree


def add_neighbour_lists(df, max_dist=15):
    nb_df = {
        f'nb_particles_{max_dist}' : np.array([None, ] * len(df)).astype(np.float64), 
        f'nb_disp_{max_dist}' : np.array([np.nan, ] * len(df)).astype(np.float64), 
        f'pid' : df['pid'].values
    }
    nb_df = pd.DataFrame(nb_df).set_index('pid')
    files = pd.unique(df['path'])
    frames = pd.unique(df['frame'])
    n_iter = len(files) * len(frames)
    with tqdm(total=n_iter, desc='Adding neighbour lists') as progress:
        for f in files:
            file_df = df[df['path'] == f]
            frames = pd.unique(df['frame'])
            for frame in frames:
                frame_wise_neigbour_lists(file_df, frame, max_dist, nb_df)
                progress.update(1)
    df = pd.concat([df.set_index('pid'), nb_df], axis=1).reset_index()
    return df


def frame_wise_neigbour_lists(df, frame, max_dist, nb_df):
    f_df = df[df['frame'] == frame]
    f_df = f_df.set_index('pid') 
    idxs = f_df.index.values
    # get the coodinates as an array (r = points, c = dims)
    p_coords = f_df[['x_s', 'ys', 'zs']].values
    tree1 = cKDTree(p_coords, copy_data=True)
    tree2 = cKDTree(p_coords, copy_data=True)
    sdm = tree1.sparse_distance_matrix(tree2, max_dist)
    array = sdm.toarray()
    for p in range(array.shape[0]):
        # get the non zero values for the platelet
        p_dists = array[p, :]
        p_idxs = np.where(p_dists > 0) # indicies to find pid
        keep = idxs[p_idxs] # pids that should be kept
        disps = list(p_dists[p_idxs])
        # add to the data frame
        pidx = idxs[p]
        nb_ps = list(f_df.loc[keep, 'particle'])
        nb_df.loc[pidx, f'nb_particles_{max_dist}'] = str(nb_ps)
        nb_df.loc[pidx, f'nb_disp_{max_dist}'] = str(disps)


# ----------------------
# Local Kinetic Measures
# ----------------------
    
#TODO

def local_contraction(df, r=15):
    '''
    Local contraction is the extent to which the platelet has moved closer to its
    assigned neighbour platelets since the previous point in time

    * sum over i = 0, ..., n
    local contraction = 
        1/n * sum(
            (x_p(t) - x_i(t))**2 + 
            (y_p(t) - y_i(t))**2 + 
            (z_p(t) - x_i(t))**2 ) ** 0.5)
        - 1/n * sum(
            (x_p(t + 1) - x_i(t + 1))**2 + 
            (y_p(t + 1) - y_i(t + 1))**2 + 
            (z_p(t + 1) - x_i(t + 1))**2 ) ** 0.5)
    '''
    files = pd.unique(df['path'])
    n_iter = len(df)
    client = Client()
    print(client.dashboard_link)
    dfs = [df[df['path'] == f].reset_index() for f in files]
    #with tqdm(total=n_iter, desc='Adding neighbour contraction') as progress:
    filewise_local_contraction = _filewise_local_contraction(r)
    df = df.set_index('pid')
    big_future = client.scatter(dfs) 
    fs = client.map(filewise_local_contraction, big_future)
    for future, result in as_completed(fs, with_results=True):
        df.loc[result[0], f'nb_cont_{r}'] = result[1]
    client.close()
    df = df.reset_index()
    return df



@curry
def _filewise_local_contraction(r,  f_df):
    #f_df = df[df['path'] == f].reset_index()
    #f_df = dd.from_pandas(f_df, 1)
    get_contraction = _local_contraction(f_df, r)
    idxs = f_df['pid'].values
    conts = []
    fn = f_df.loc[0, 'path']
    #conts = f_df['pid'].map(get_contraction, meta=pd.Series({f'nb_cont_{r}' : []}, dtype=np.float64))
    with tqdm(total=len(f_df), desc=f'Local contraction: {fn}') as progress:
        for p in idxs:
            cont = get_contraction(p)
            conts.append(cont)
            progress.update(1)
        #df.loc[idxs, f'nb_cont_{r}'] = conts
        #print('file done')
    return idxs, conts


@curry
def _local_contraction(df, r, pid):
    # apply this to df grouped by path (i.e., single video file)
    row = df[df['pid'] == pid].set_index('pid')
    nbs = row.loc[pid, f'nb_particles_{r}']
    nbs = _ensure_list(nbs)
    nb_disp = row.loc[pid, f'nb_disp_{r}']
    nb_disp = _ensure_list(nb_disp)
    frame = row.loc[pid, 'frame'] 
    pt = row.loc[pid, 'particle']
    # find the location of the platelet at the next point in time
    if row.loc[pid, 'nrtracks'] > row.loc[pid, 'tracknr']:
        tp1_row = df[(df['particle'] == pt) & (df['frame'] == frame + 1)].reset_index() # would have to include path to generalise
        inc = 1
        if len(tp1_row) == 0:
            tp1_row = df[(df['particle'] == pt) & (df['frame'] == frame + 2)].reset_index()
            inc = 2
        if len(tp1_row) > 0:
            tp1_coords = np.array([tp1_row.loc[0, 'x_s'], tp1_row.loc[0, 'ys'], tp1_row.loc[0, 'zs']])
            # find the location of the current neighbours at the next point in time
            nb_coords = []
            for nb in nbs:
                nb_row = df[(df['particle'] == nb) & (df['frame'] == frame + 1)].reset_index() # would have to include path to generalise
                if len(nb_row) > 0:
                    coords = np.array([nb_row.loc[0, 'x_s'], nb_row.loc[0, 'ys'], nb_row.loc[0, 'zs']])
                    nb_coords.append(coords)
            if len(nb_coords) > 0:
                nb_coords = np.stack(nb_coords, axis=0)
                tp1_diff = tp1_coords - nb_coords
                tp1_disp = np.linalg.norm(tp1_diff, axis=1) # compute 2-norm for each row vector
                local_cont = np.mean(nb_disp) - (np.mean(tp1_disp)/inc)
            else: 
                local_cont = np.NaN
        else: 
            local_cont = np.NaN
    else: 
        local_cont = np.NaN
    return local_cont


# ---------------------
# Local Static Measures
# ---------------------

def local_density(df, r=15, z_max=66):
    '''
    Density of the neighbour platelets in the sphere (radius r) around the platelet. 
    The area of the sphere is corrected for the top and the bottom of the image 

    Define the height h above or below the image:
    h = r - z             if z <= r
    h = 0                 if z < z + r < 66 
    h = z + r - 66        if z + r > 66
    where z is the height of the platelet in z (um)

    A = (4/3)πr**3 - (1/3)πh**2(3r -h)
    neighbour density = n/A
    '''
    df = df.set_index('pid')
    files = pd.unique(df['path'])
    n_iter = len(df)
    sphere_size = ((4 * np.pi * r**3) / 3)
    with tqdm(total=n_iter, desc='Adding neighbour density') as progress:
        for f in files:
            f_df = df[df['path'] == f].reset_index()
            idxs = f_df['pid'].values
            get_density = _local_density(f_df, r, z_max, sphere_size)
            #densities = f_df['pid'].apply(get_density)
            densities = []
            for p in idxs:
                density = get_density(p)
                densities.append(density)
                progress.update(1)
            df.loc[idxs, f'nb_density_{r}'] = densities
    return df



@curry
def _local_density(df, r, z_max, sphere_size, pid):
    row = df[df['pid'] == pid].set_index('pid')
    nbs = row.loc[pid, f'nb_particles_{r}']
    nbs = _ensure_list(nbs)
    z = row.loc[pid, 'zs']
    if z <= r:
        h = r - z
        area = sphere_size - ((np.pi * h**2) * (3 * r - h) / 3)
    elif (z + r) > z_max:
        h = z + r - z_max
        area = sphere_size - ((np.pi * h**2) * (3 * r - h) / 3)
    else:
        area = sphere_size
    density = len(nbs) / area
    return density



# ----------------------
# Local Dynamic Measures
# ----------------------




# ---------------
# Embolism Events
# ---------------

def embolysis(df, r_emb=10):
    emb_df = {
        'emb_nbs' : [None, ] * len(df), 
        'pid' : df['pid'].values 
    }
    emb_df = pd.DataFrame(emb_df).set_index('pid')
    df['terminates'] = df['tracknr'] == df['nrtracks']
    term_df = df[df['terminates'] == True]
    files = pd.unique(df['path'])
    get_embolysis_freinds = _embolysis_partners(df, r_emb)
    for f in files:
        p_df = term_df[term_df['path'] == f]
        idxs = p_df['pid'].values # not index for p_df but for emb_df
        emb_ptns = p_df['pid'].apply(get_embolysis_freinds)
        emb_df.loc['emb_nbs', idxs] = emb_ptns
    df = pd.concat([df.set_index('pid'), emb_df], axis=1).reset_index()  
    return df


@curry
def _embolysis_partners(df, r_emb, pid):
    row = df[df['pid'] == pid].reset_index()
    # are any neighbours also terminating at this time point or in the next?
    nbs = row.loc['nb_particles', 0]
    nbs = _ensure_list(nbs)
    nb_disp = row.loc['nb_disp', 0]
    nb_disp = _ensure_list(nb_disp)
    frame = row.loc['frame', 0]
    nb_ptns = []
    for i, n in enumerate(nbs):
        if nb_disp[i] <= r_emb:
            nb_row = df[df['particle'] == n].reset_index()
            if nb_row.loc['terminating', 0]:
                nb_ptns.append(n)
            else: # only need to move to next tp if not terminating
                nb_row = df[(df['particle'] == n) & (df['frame'] == frame + 1)]
                if nb_row.loc['terminating', 0]:
                    nb_ptns.append(n)
    if len(nb_ptns) == 0:
        nb_ptns = None
    return nb_ptns


def find_emboli(df):
    """
    An algorithm for finding individual embolysis events and labeling with a 
    unique id number. Embolysis partners must first be found. 
    """
    # get the terminating platelet entries (i.e., terminates as of next frame)
    term_df = df[df['terminates'] == True].set_index('pid')
    term_df = term_df.sorted('frame') # sort according to frame
    # Seed 0 given to the first row and any of its partners
    # continue spreading the seed value to new partners recursively
    # once the seed can no longer be spread, find the next value with no seed
    # assigned and increment seed by 1. Continue until each platelet is assigned to an
    # embolism event



# -----------------
# Utility Functions
# -----------------

def _ensure_list(list_or_string):
    if isinstance(list_or_string, str):
        nbs = eval(list_or_string)
    else:
        nbs = list_or_string
    assert isinstance(nbs, list)
    return nbs


# use this in a loop in which the path is specified
# i.e., df = bigger_df[bigger_df['path'] == <a path>]
def get_neighbour_df(df, p, max_dist):
    p_row = df[df['pid'] == p].reset_index()
    #path = p_row.loc[0, 'path']
    #frame = p_row.loc[0, 'frame']
    nbs = eval(p_row.loc[0, f'nb_particles_{max_dist}'])
    for n in nbs:
        sml_df = ...