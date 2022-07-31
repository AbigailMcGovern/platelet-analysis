from multiprocessing.dummy import freeze_support
from tkinter import X
from plateletanalysis.variables.neighbours import local_contraction
import os
import pandas as pd

if __name__ == '__main__':
    freeze_support()
    d = '/fs02/rl54/dataframes/'
    p = os.path.join(d, '211206_saline_df_220611-0.parquet')
    df = pd.read_parquet(p)

    try:
        df = df.drop(columns=['level_0'])
        df = df.drop(columns=['index'])
    except:
        pass


    p = os.path.join(d, '211206_saline_df_220612-0.parquet')
    df = local_contraction(df)
    df.to_parquet(p)

