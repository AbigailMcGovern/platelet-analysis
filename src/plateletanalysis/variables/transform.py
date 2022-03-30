import numpy as np
from scipy.spatial.transform import Rotation as Rot
import pandas as pd
from toolz import curry

# ---------------------
# Cartesian Coordinates
# ---------------------

def adjust_coordinates(
    df, 
    meta_df, 
    rot_angle=45, 
    file_col='file', 
    roi_x_col='roi_x', 
    roi_y_col='roi_y', 
    path_col='path', 
    xs_col='x_s', 
    ys_col='ys',
    zs_col='zs',
    px_microns_col=None, 
    ):
    '''
    Adjust the platelet coordinates such as to aline the x-axis with the 
    axis of blood flow.

    Check:
    - rotation is in fact counter clockwise to the xy plane
    - which direction the blood is flowing along the axis 
        (i think this should be in the positive direction)
    '''
    for index, row in meta_df.iterrows():
        file=row[file_col]
        # if pixel microns is supplied, scale ROIs, if not, don't
        if px_microns_col is not None:
            px_microns = row[px_microns_col]
        else:
            px_microns = 1
        adjust_x=row[roi_x_col] * px_microns
        adjust_y=row[roi_y_col] * px_microns
        xs_1=df.loc[df[path_col]==file, xs_col].copy()
        ys_1=df.loc[df[path_col]==file, ys_col].copy()
        #print(xs_1,xs_1-adjust_x)
        #zs_1=df_exp.loc[df_exp.file==file,'zs'].copy()
        df.loc[df.path==file, xs_col]=xs_1-adjust_x#+150
        df.loc[df.path==file, ys_col]=ys_1-adjust_y
    rot = Rot.from_euler('z', -rot_angle, degrees=True)#Förut -rot_angle
    xyz = df[[xs_col, ys_col, zs_col]].to_numpy()
    xyz_rot = rot.apply(xyz)
    df[xs_col], df[ys_col] = xyz_rot[:,0], xyz_rot[:,1]
    return df # superflous return... just to make more readable when used *upside down smiley*


def z_floor(
    df, 
    zs_col='zs', 
    ):
    '''
    Calls the function zfloor with PC sorted by path with index set to 'pid', 
    returns a data frame with a column 'zf'
    '''
    z_floor = _z_floor(zs_col) # curried version of z_floor with zs_col parameter set
    z_grp = df.set_index('pid').groupby(['path']).apply(z_floor).reset_index()
    df = pd.concat([df.set_index('pid'), z_grp.set_index('pid')], axis=1).reset_index()
    return df


@curry
def _z_floor(
    zs_col, 
    pc,
    ):
    
    '''
    subtracts the height of the second percentile of the platelets after
    z-pos from z-pos for each platelet and adds this value as zf
     '''
    df=pc
    # Sw: Tar ut frame 1-29. 
    # Eng: Remove frames 1-29
    #   (does frame indexing start @ 0 or 1?)
    df=df[df.frame<30]
    # Sw: Definierar andra percentilen som botten pa proppen
    # Eng: Defines the second percentile as the bottom of the plug
    floor=np.percentile(df[zs_col], 2)
    # Sw: Subtraherar detta varde fran z-pos
    # Eng: Subtract this value from z-pos
    zf=pc[zs_col]-floor
    return pd.DataFrame({'zf' : (zf)})


# -----------------
# Polar coordinates
# -----------------

def spherical_coordinates(df):
    '''
    Convert cartesian coordinates to a spherical coordinate system (which is 
    equavalent to 2D polar coordinates). In this system, the coordinates are
    expressed as (rho, theta, phi) rather than (x, y, z). Here, rho represents
    the magnetude of the vector from the origin, phi represents the angle of
    the vector from the x-axis when projected onto the x-y plane, and theta 
    represents the angle of the vector in from z along a plane that runs 
    parallel to the vector and orthoganal to the x-y plane. 

    More formally:
    rho = (x ** 2 + y ** 2 + z ** 2) ** 0.5
    theta = arccos(z / rho)
    phi = arctan(y / x)

    In this case, in order to make the values of phi optimally useful, 
    phi has been adjusted such that phi = arctan(abs(y) / x). 
    e.g., coords = (34, -78, 30), phi = arctan(78 / 34) = 1.16 (~ 66.4˚)

    Units:
    rho - um
    theta - radians (0 ≤ theta ≤ π/2)
    phi - radians (-π/2 ≤ theta ≤ π/2)

    Rho represents displacement of the platelet from the 
    centre of the injury. Theta represents the position in z according to the 
    platelet's height in the sphere. Larger values of theta represent platelets
    lower in the dome. Phi represents the position of platelets in the xy
    plane. 0-90˚ (0-π/2) represents platelets in the anterior of the clot and 
    -90˚-0 (-π/2-0) represents platelets in the posterior. Therefore, the 
    coordinates are insensitive to which side of the clot the platelet is 
    found on, only at what angle the platelet is at relative to the blood flow.
    '''
    rho = _spherical_coord_rho(df)
    df = pd.concat([df.set_index('pid'), rho.set_index('pid')], axis=1).reset_index()
    theta = _spherical_coord_theta(df)
    phi = _spherical_coord_phi(df)
    df = pd.concat([df.set_index('pid'), theta.set_index('pid'), phi.set_index('pid')], axis=1).reset_index()
    return df


def _spherical_coord_rho(df):
    rho = np.sqrt((df['x_s'].values ** 2) + (df['ys'].values ** 2) + (df['zs'].values ** 2))
    return pd.DataFrame({'pid' : df['pid'].values, 'rho' : rho})


def _spherical_coord_theta(df):
    theta = np.arccos(df['zs'].values / df['rho'].values)
    return pd.DataFrame({'pid' : df['pid'].values, 'theta' : theta})


def _spherical_coord_phi(df):
    phi = np.arctan(df['ys'].values / df['x_s'].values)
    pd.DataFrame({'pid' : df['pid'].values, 'phi' : phi})