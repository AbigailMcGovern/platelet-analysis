from multiprocessing.dummy import freeze_support
from plateletanalysis.variables.neighbours import local_contraction
import os
import pandas as pd

if __name__ == '__main__':
    freeze_support()
    d = '/fs02/rl54/dataframes/'
    p = os.path.join(d, '220603_211206_biva_df_spherical-coords.parquet')
    df = pd.read_parquet(p)


    p = os.path.join(d, '211206_biva_df_220612-0.parquet')
    df = local_contraction(df)
    df.to_parquet(p)

