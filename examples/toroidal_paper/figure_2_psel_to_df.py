import pandas as pd
from plateletanalysis.variables.basic import add_nrtracks, tracknr_variable, time_seconds
from plateletanalysis.variables.transform import adjust_coordinates
from plateletanalysis.variables.neighbours import add_neighbour_lists, local_density
from plateletanalysis.variables.measure import quantile_normalise_variables_frame, finite_difference_derivatives
import os

d = '/Users/abigailmcgovern/Data/platelet-analysis/P-selectin/P-selectin'
n = '230117_110631_segtrack_segmentation-metadata.csv'
mp = os.path.join(d, n)

files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith('tracks.csv')]

df = [pd.read_csv(p) for p in files]
df = pd.concat(df).reset_index(drop=True)
mdf = pd.read_csv(mp)

sp ='/Users/abigailmcgovern/Data/platelet-analysis/dataframes/230919_p-selectin.parquet'


# Basic variables
# ---------------
renaming  = {
    'xs' : 'x_s', 
    'file' : 'path', 
    'elongation' : 'elong', 
    't' : 'frame'
}
df = df.rename(columns=renaming)

df = add_nrtracks(df)
df = tracknr_variable(df)
df = time_seconds(df)
df = adjust_coordinates(df, mdf)
df.to_parquet(sp)


# Velocity
# --------
df = finite_difference_derivatives(df)
df.to_parquet(sp)


# Local density
# -------------
df = add_neighbour_lists(df)
df = local_density(df)
df = quantile_normalise_variables_frame(df)
df.to_parquet(sp)




