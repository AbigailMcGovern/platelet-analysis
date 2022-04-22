import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from toolz import curry


# We want to do this using a CKD tree... the current implementation is massively inefficient 


def add_neighbour_lists(df, max_dist=15):
    nb_df = {
        f'nb_particles_{max_dist}' : np.array([np.nan, ] * len(df)).astype(np.float64), 
        f'nb_disp_{max_dist}' : np.array([np.nan, ] * len(df)).astype(np.float64), 
        f'pid' : df['pid'].values
    }
    nb_df = pd.DataFrame(nb_df).set_index('pid')
    files = pd.unique(df['path'])
    for f in files:
        file_df = df[df['path'] == f]
        frames = pd.unique(df['frame'])
        for frame in frames:
            frame_wise_neigbour_lists(file_df, frame, max_dist, nb_df)
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
    for p in idxs:
        # get the non zero values for the platelet
        p_dists = array[p, :]
        p_idxs = np.where(p_dists > 0) # indicies to find pid
        keep = idxs[p_idxs] # pids that should be kept
        disps = list(p_dists[p_idxs])
        # add to the data frame
        nb_ps = list(f_df['particle'].values[keep])
        nb_df.loc['nb_particles', p] = nb_ps
        nb_df.loc['nb_disp', p] = disps


    
#TODO

def local_contraction(df):
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
    #TODO: finish implementing this, the computation has been worked out below
    pass


@curry
def _local_contraction(df, pid):
    # apply this to df grouped by path (i.e., single video file)
    row = df[df['pid'] == pid].set_index('pid')
    nbs = row.loc['nb_particles', pid]
    nbs = _ensure_list(nbs)
    nb_disp = row.loc['nb_particles', pid]
    nb_disp = _ensure_list(nb_disp)
    frame = row.loc['frame', pid] 
    pt = row.loc['particle', pid]
    # find the location of the platelet at the next point in time
    tp1_row = df[(df['particle'] == pt) & (df['frame'] == frame + 1)].reset_index() # would have to include path to generalise
    tp1_coords = np.array([tp1_row.loc['x_s', 0], tp1_row.loc['ys', 0], tp1_row.loc['zs', 0]])
    # find the location of the current neighbours at the next point in time
    nb_coords = []
    for nb in nbs:
        nb_row = df[(df['particle'] == nb) & (df['frame'] == frame + 1)].reset_index() # would have to include path to generalise
        coords = np.array([nb_row.loc['x_s', 0], nb_row.loc['ys', 0], nb_row.loc['zs', 0]])
        nb_coords.append(coords)
    nb_coords = np.stack(nb_coords, axis=0)
    tp1_diff = tp1_coords - nb_coords
    tp1_disp = np.linalg.norm(tp1_diff, axis=1) # compute 2-norm for each row vector
    local_cont = np.mean(nb_disp) - np.mean(tp1_disp)
    return local_cont




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
    get_density = _local_density(df, r, z_max)
    densities = df['pid'].apply(get_density)
    df[f'nb_density_{r}'] = densities
    return df



@curry
def _local_density(df, r, z_max, pid):
    row = df[df['pid'] == pid].set_index('pid')
    nbs = row.loc['nb_particles', pid]
    nbs = _ensure_list(nbs)
    z = row.loc['zs', pid]
    if z <= r:
        h = r - z
    elif (z + r) < z_max and (z - r) > 0:
        h = 0
    elif (z + r) > z_max:
        h = z + r - z_max
    area = ((4 * np.pi * r**3) / 3) - ((np.pi * h**2) * (3 * r - h) / 3)
    density = len(nbs) / area
    return density


def _ensure_list(list_or_string):
    if isinstance(list_or_string, str):
        nbs = eval(list_or_string)
    assert isinstance(list_or_string, list)
    return list_or_string
