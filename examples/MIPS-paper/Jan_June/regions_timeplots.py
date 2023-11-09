import pandas as pd
from plateletanalysis import regions_abs_and_pcnt_timeplots

# NOTE: Run this in the plotting envrionment. 

# ------------------------------
# Some of the possible variables
# ------------------------------
names_pcnt = {
        0 :'platelet count pcnt'  ,              
        1 :'platelet density (um^-3) pcnt'  ,  #  1
        2 :'thrombus edge distance (um) pcnt' ,  
        3 :'recruitment (s^-1) pcnt'     ,      #  3
        4 :'shedding (s^-1) pcnt'   ,            
        5 :'mean stability pcnt'    ,            
        6 :'mean tracking time (s) pcnt' ,       
        7 :'sliding (ums^-1) pcnt'         ,     
        8 :'proportion < 15 s pcnt'       ,      
        9 :'proportion > 60 s pcnt'      ,      
        10 :'tracking time IQR (s) pcnt'   ,     
        11 :'proportion shed < 15 s pcnt'    ,    
        12 :'proportion shed > 60 s pcnt'    ,    
        13 :'proportion recruited < 15 s pcnt' ,  # 13
        14 :'proportion recruited > 60 s pcnt' ,  
    }
names = {
        0 :'platelet count'  ,              
        1 :'platelet density (um^-3)'  ,    
        2 :'thrombus edge distance (um)' ,  
        3 :'recruitment (s^-1)'     ,       
        4 :'shedding (s^-1)'   ,            
        5 :'mean stability'    ,            
        6 :'mean tracking time (s)' ,       
        7 :'sliding (ums^-1)'         ,     
        8 :'proportion < 15 s'       ,      
        9 :'proportion > 60 s'      ,      
        10 :'tracking time IQR (s)'   ,     
        11 :'proportion shed < 15 s'    ,    
        12 :'proportion shed > 60 s'    ,    
        13 :'proportion recruited < 15 s' ,  
        14 :'proportion recruited > 60 s' ,  
    }

# -------------
# Customisables
# -------------

do_counts = True

# -----
# Plots
# -----

if do_counts:
    p = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/regions_data/230420_regionsdata_9var_trk1_seconds.csv'
    data = pd.read_csv(p)
    regions_abs_and_pcnt_timeplots(data, (names[0], names_pcnt[0],), time_col='time (s)', 
                      errorbar=False, treatements=(('MIPS', 'DMSO (MIPS)'), ('MIPS', 'SQ', 'cangrelor')), 
                      different_treatements=True) 
