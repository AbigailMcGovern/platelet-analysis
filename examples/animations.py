from matplotlib.pyplot import get
import napari
import zarr
from plateletanalysis.visualise.tracks import read_ND2, get_tracks
import pandas as pd
import os
from skimage.measure import regionprops_table
from skimage.util import map_array
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm


# images (calcium, fibrin, GPIIb, TD)
d = '/Users/amcg0011/Data/platelet-analysis/demo-data'
ip = os.path.join(d, '210520_IVMTR108_Inj3_MIPS_exp3.nd2')
images = read_ND2(ip)
images = images[:3] # remove the empty TD channel

# segmentation
sp = os.path.join(d, '210920_141056_seg-track_210520_IVMTR108_Inj3_MIPS_exp3_labels.zarr')
seg = zarr.open(sp)

# tracks
tp = os.path.join(d, '210920_141056_seg-track_210520_IVMTR108_Inj3_MIPS_exp3_platelet-coords_tracks.csv')
tdf = pd.read_csv(tp)
tdf = tdf[tdf['track_no_frames'] > 2]
tracks = get_tracks(tdf, ('particle', 't', 'z_pixels', 'y_pixels', 'x_pixels'))

# remove untracked segments
def keep_only_tracked_segments(tracks, seg):
    for t in np.unique(tracks[:, 1]):
        t = int(t)
        t_kdtree = cKDTree(tracks[:, 2:])
        new_seg = np.zeros_like(seg)
        props_df = regionprops_table(seg[t, ...], properties=('label', 'coords'))
        coords = props_df['coords']
        for c_list in coords:
            c_kdtree = cKDTree(c_list)
            n = t_kdtree.count_neighbors(c_kdtree, r=1)
            if n > 0:
                idx = (t, c_list[:, 0], c_list[:, 1], c_list[:, 2])
                new_seg[idx] = 1
    return new_seg


def only_tracked_segments(df, seg):
    new_seg = np.zeros_like(seg)
    for t in tqdm(range(seg.shape[0])):
        tdf = df[df['t'] == t]
        labs = tdf['label'].to_numpy()
        map_array(np.asarray(seg[t]), labs, labs, out=new_seg[t])
    return new_seg




clean_seg = only_tracked_segments(tdf, seg)
#cp = os.path.join(d, '210920_141056_seg-track_210520_IVMTR108_Inj3_MIPS_exp3_labels_clean.zarr')
#zarr.save(cp, clean_seg)

# display the data

v = napari.view_image(images[0], scale=(1, 4, 1, 1), name='Ca2+', blending='additive', colormap='green')
v.add_image(images[1], scale=(1, 4, 1, 1), name='Fibrin', blending='additive', colormap='red')
v.add_image(images[2], scale=(1, 4, 1, 1), name='GPIIB', blending='additive', colormap='magenta')
v.add_labels(clean_seg, scale=(1, 4, 1, 1))
v.add_labels(seg, scale=(1, 4, 1, 1))
v.add_tracks(tracks, scale=(1, 4, 1, 1))

napari.run()

