import pandas as pd
import numpy as np
from plateletanalysis.variables.transform import cylindrical_coordinates

 

def average_trajectories(
        df, 
        frame=35, 
        centre_dists=((0, 25), (25, 50), (50, 75), (75, 100)), 
        centre_col='dist_c_pcnt',
        phi_angs=((0, 0.5 * np.pi), (- 0.5 * np.pi, 0)), 
        phi_col='phi',
        theta_angs=((0, 0.5 * np.pi), (0.5 * np.pi, np.pi)), 
        theta_col='theta',
        n_frames=20)
    :
    df = cylindrical_coordinates(df)
    r_col = 'cyl_radial'
    z_col = 'zs'
    f0 = frame - n_frames - 1
    df = df[(df['frame'] > f0) & (df['frame'] < frame)]
    fmax = df['frame'].max()
    final_df = df[df['frame'] == fmax]
    traj_df = {
        'particle': [], 
        'frame' : [], 
        r_col : [],
        z_col : [], 
        'final_position' : [],
        'centre_dist' : [], 
        'final_centre_dist' : [], 
        'final_phi_angs' : [], 
        'final_theta_angs' : []
    }
    particles = pd.unique(df['particle'])
    for p in particles:
        pdf = df[df['particle'] == p]
        if len(pdf >= n_frames)
        fpdf = final_df[final_df['particle'] == p]
        pos_str, pos = _position_string(fpdf, centre_dists, phi_angs, theta_angs, centre_col, phi_col, theta_col)
        final_pos = fpdf[[centre_col, phi_col, theta_col]].values




def _position_string(fpdf, centre_dists, phi_angs, theta_angs, centre_col, phi_col, theta_col):
    for cd in centre_dists:
        if fpdf[centre_col] > cd[0] & fpdf[centre_col] < cd[1]:
            cd_cat = cd
    for phi in phi_angs:
        if fpdf[phi_col] > phi[0] & fpdf[phi_col] < phi[1]:
            phi_cat = phi
    for theta in theta_angs:
        if fpdf[theta_col] > theta[0] & fpdf[theta_col] < theta[1]:
            theta_cat = theta
    pos_str = f'CD{cd_cat}-phi{phi_cat}-theta{theta_cat}'
    return pos_str, (cd, phi, theta)
    

    
    