import numpy as np
from scipy.spatial.transform import Rotation as Rot
import pandas as pd
from toolz import curry
from plateletanalysis.analysis.peaks_analysis import var_over_cylr
from collections import defaultdict

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
        adjust_y=row[roi_x_col] #* px_microns
        adjust_x=row[roi_y_col] #* px_microns
        xs_1=df.loc[df[path_col]==file, xs_col].copy()
        ys_1=df.loc[df[path_col]==file, ys_col].copy()
        #print(xs_1,xs_1-adjust_x)
        #zs_1=df_exp.loc[df_exp.file==file,'zs'].copy()
        df.loc[df[path_col] == file, xs_col]=xs_1-adjust_x#+150
        df.loc[df[path_col] == file, ys_col]=ys_1-adjust_y
    rot = Rot.from_euler('z', -rot_angle, degrees=True)#Förut -rot_angle
    xyz = df[[xs_col, ys_col, zs_col]].to_numpy()
    xyz_rot = rot.apply(xyz)
    df[xs_col], df[ys_col] = xyz_rot[:,0], xyz_rot[:,1]
    df[ys_col] = - df[ys_col]
    return df # superflous return... just to make more readable when used *upside down smiley*



def revert_to_pixel_coords(df, meta_dfs): 
    for meta in meta_dfs:
        p = meta.loc[0, 'file']
        fdf = df[df['path'] == p]
        idxs = fdf.index.values

        # inverse the rotation (+45 degrees)
        rot = Rot.from_euler('z', 45, degrees=True)
        xyz = fdf[['x_s', 'ys', 'zs']].to_numpy()
        xyz_rot = rot.apply(xyz)
        x_pixels = xyz_rot[:, 0]
        y_pixels = xyz_rot[:, 1]
        z_pixels = fdf['zs'].values
    
        # add the adjustment value (was initially removed befor rotation)
        scale = eval(meta.loc[0, 'scale'])
        adjust_x = meta.loc[0, 'roi_x'] #* scale[3]
        adjust_y = meta.loc[0, 'roi_y'] #* scale[2]
        x_pixels = x_pixels + adjust_x 
        y_pixels = y_pixels + adjust_y
    
        # swap x & y
        x = y_pixels
        y = x_pixels
    
        # divide by the scale
        x = x / scale[3]
        y = y / scale[2]
        z_pixels = z_pixels / scale[1]

        # final 180 degree rotation ... unsure why this is necessary... just works
        rot = Rot.from_euler('z', 180, degrees=True)
        xyz = np.stack([x, y, z_pixels], axis=1)
        xyz_rot = rot.apply(xyz)

        # assign values into original df
        df.loc[idxs, 'x_pixels'] = xyz_rot[:, 0]
        df.loc[idxs, 'y_pixels'] = xyz_rot[:, 1]
        df.loc[idxs, 'z_pixels'] = z_pixels

    return df




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
    theta = 1/2 * π - arccos(z / rho)
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
    lower higher in the dome. Phi represents the position of platelets in the xy
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
    theta = 0.5 * np.pi - theta # this changes the angle so that 0 rad when platelet at bottom of image
    return pd.DataFrame({'pid' : df['pid'].values, 'theta' : theta})


def _spherical_coord_phi(df):
    phi = np.arctan(df['ys'].values / np.abs(df['x_s'].values))
    return pd.DataFrame({'pid' : df['pid'].values, 'phi' : phi})



# -----------------------
# Cylindrical coordinates
# -----------------------


def cylindrical_coordinates(df):
    '''
    Find cylindircal coordinates (rho, phi, z - there are many naming conventions 
    so i picked the one most consistent with how I named the spherical coords... 
    doesn't matter anyway, the columns are named as radial [rho] and azimuthal
    [phi]).

    This is used to project onto a place sweeping through the donut ring. 
    Do this by using just the radial and z coordinates. You may or may not need
    to quantile normalise the radial coordinate in sectors to account for 
    differences in thickness in different parts of the clot.
    '''
    radial = (df['x_s'].values ** 2 + df['ys'].values ** 2 ) ** 0.5
    df['xyr'] = list(zip(list(df['x_s'].values), list(df['ys'].values), list(radial)))
    azimuthal = df['xyr'].apply(_calculate_cylindrical_azimuthal)
    df = df.drop('xyr', axis=1) 
    df['cyl_radial'] = radial
    df['cyl_azimuthal'] = azimuthal
    return df


def _calculate_cylindrical_azimuthal(xyr):
    # visualise and adjust to suit needs
    '''In degrees not radians'''
    x, y, r = xyr
    phi = np.arctan(y/np.abs(x)) * 180 / np.pi
    return phi



# --------------------
# Toroidal coordinates
# --------------------

def toroidal_coordinates(df):
    """
    A coordinate system that offers a thrombus specific adjustment of the 
    """
    # get information about max density peak distance from centre
    gb = ['path', 'cylr_bin', 'time_bin']
    #data = var_over_cylr(df, 'nb_density_15', gb)
    #data_max = path_dist_at_max(data, ['path', 'time_bin'], 'nb_density_15', dist='cylr_bin')
    # get the tor rho coordinate
    #df = tor_rho_coord(data_max, df) # tor has three coords: rho, z, and theta
    df['tor_rho'] = df['cyl_radial'] - 37.5
    # get tor theta coordinate: will go between +90 (floor outer edge) and -90 (floor inner edge)
    df = tor_theta_coord(df)
    # the coordinates have been visually checked... looks very good!! 
    return df


def path_dist_at_max(data, gb, col, dist='cylr_bin'):
    # return df of path and max
    out = defaultdict(list)
    for k, grp in data.groupby(gb):
        for i, c in enumerate(gb):
            out[c].append(k[i])
        midx = np.argmax(grp[col].values)
        m = grp[col].values[midx]
        md = grp[dist].values[midx]
        out[col].append(m)
        out[dist].append(md)
    out = pd.DataFrame(out)
    return out


def tor_rho_coord(data_max, df):
    for k, grp in df.groupby(['path', 'time_bin']):
        pdata = data_max[(data_max['path'] == k[0]) & (data_max['time_bin'] == k[1])]
        dist_max = pdata['cylr_bin'].values
        idxs = grp.index.values
        vals = grp['cyl_radial'] - dist_max
        df.loc[idxs, 'tor_rho'] = vals
    return df


def tor_theta_coord(df):
    # in degrees 
    # more negative = more central/epithelial
    # more positive = more distal/epithelial
    tan_theta = df['zs'] / df['tor_rho']
    coef = df['tor_rho'] / np.abs(df['tor_rho'])
    df['tor_theta'] = coef * 90 - np.arctan(tan_theta) / np.pi * 180
    return df


