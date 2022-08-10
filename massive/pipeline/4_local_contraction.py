import pandas as pd
import argparse
from plateletanalysis.variables.neighbours import local_contraction
from multiprocessing.dummy import freeze_support


if __name__ == '__main__':
    freeze_support()

    p = argparse.ArgumentParser()
    p.add_argument('-f', '--file', help='parquet file platelet info')
    p.add_argument('-s', '--save', help='parquet file into which to save output')
    args = p.parse_args()
    
    p = args.file
    sp = args.save
    df = pd.read_parquet(p)
    
    df = local_contraction(df)
    
    df.to_parquet(sp)
