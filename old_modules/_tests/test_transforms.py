from audioop import add
from plateletanalysis.variables.transform import spherical_coordinates
import numpy as np
from pathlib import Path
import os 
import pandas as pd


dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
pa_dir = dir_path.parents[0] # plateletanalysis directory

def read_data():
    data_path = os.path.join(pa_dir, 'data', 'test_data', 'test_data.parquet') # path to test data 
    # test data is frames 0-1 for 191016_IVMTR12_Inj1_ctrl_exp3
    df = pd.read_parquet(data_path)
    return df

def test_spherical_coordinates():
    df = read_data()
    df = spherical_coordinates(df)
    # assert
    return df

if __name__ == '__main__':
    from plateletanalysis.variables.measure import add_finite_diff_derivative
    df = test_spherical_coordinates()

    print('rho max: ', df['rho'].max(), ' um')
    print('rho min: ', df['rho'].min(), ' um')
    print('phi max: ', df['phi'].max(), ' radians')
    print('phi min: ', df['phi'].min(), ' radians')
    print('theta max: ', df['theta'].max(), ' radians')
    print('theta min: ', df['theta'].min(), ' radians')

    df = add_finite_diff_derivative(df, 'rho')
    df = add_finite_diff_derivative(df, 'phi')
    df = add_finite_diff_derivative(df, 'theta')

    #data_path = os.path.join(pa_dir, 'data', 'test_data', 'test_data.parquet')
    #df.to_parquet(data_path)