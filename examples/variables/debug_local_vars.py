from multiprocessing.dummy import freeze_support
from plateletanalysis.variables.neighbours import local_contraction
import pandas as pd
import os

if __name__ == '__main__':
    freeze_support()


    d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
    sp = os.path.join(d, '211206_saline_df_nb_dens.parquet')
    df = pd.read_parquet(sp)
    #p1df = df[df['path'] == '191030_IVMTR18_Inj2_saline_exp3']
    #p1df = local_contraction(p1df)

    df = local_contraction(df)
    df.to_parquet(sp)