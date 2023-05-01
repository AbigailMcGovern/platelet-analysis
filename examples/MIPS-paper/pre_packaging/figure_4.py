import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from plateletanalysis.variables.basic import add_basic_variables_to_files



def regions_bar_charts(
        df, 
        peaks,
        var, 
        treatements=('MIPS', 'SQ', 'cangrelor'), 
        controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline'),
        regions=('center', 'anterior', 'lateral', 'posterior'), 
        groupby=('region', )
    ):
    '''
    Use with full df after applying add_basic_variables_to_files 
    (makes sure all of the necessary vars are included). 

    Parameters
    ----------
    df: pd.DataFrame
    peaks: 
    '''
    # remove unnecessary 
    df = df[df['nrtracks'] > 1] 
    for k, grp in df.groupby():
        pass