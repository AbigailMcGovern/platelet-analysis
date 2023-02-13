import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


def regions_lineplots(
        data, 
        variables,
        treatements=('MIPS', 'SQ', 'cangrelor'), 
        regions=('center', 'anterior', 'lateral', 'posterior'), 
        time_col='time (s)',
        hue='treatment', 
        errorbar=False,
    ):
    dfs = []
    for t in treatements:
        sdf = data[data['treatment'] == t]
        dfs.append(sdf)
    data = pd.concat(dfs).reset_index(drop=True)
    del dfs
    if errorbar:
        es = 'bars'
        m = True
    else:
        es = 'band'
        m = None
    fig, axs = plt.subplots(len(variables), len(regions))
    for j in range(len(regions)):
        r = regions[j]
        sdf = data[data['region'] == r]
        if len(variables) > 1:
            for i in range(len(variables)):
                ax = axs[i, j]
                ax.set_title(r)
                sns.lineplot(data=sdf, x=time_col, y=variables[i], hue=hue, ax=ax, errorbar=("se", 1), err_style=es, markers=m)
        else:
            ax = axs[j]
            ax.set_title(r)
            sns.lineplot(data=sdf, x=time_col, y=variables[0], hue=hue, ax=ax, errorbar=("se", 1), err_style=es, markers=m)
    plt.show()



if __name__ == '__main__':
    names=('platelet count pcnt', 'platelet density (um^-3) pcnt', 'thrombus edge distance (um) pcnt', 
               'recruitment (s^-1) pcnt', 'shedding (s^-1) pcnt', 'mean stability pcnt', 
               'dvy (um s^-1) pcnt', 'mean tracking time (s) pcnt')
    names_pcnt = {
        0 :'platelet count pcnt'  ,              
        1 :'platelet density (um^-3) pcnt'  ,    
        2 :'thrombus edge distance (um) pcnt' ,  
        3 :'recruitment (s^-1) pcnt'     ,       
        4 :'shedding (s^-1) pcnt'   ,            
        5 :'mean stability pcnt'    ,            
        6 :'mean tracking time (s) pcnt' ,       
        7 :'sliding (ums^-1) pcnt'         ,     
        8 :'proportion < 15 s pcnt'       ,      
        9 :'proportion > 60 s pcnt'      ,      
        10 :'tracking time IQR (s) pcnt'   ,     
        11 :'proportion shed < 15 s pcnt'    ,    
        12 :'proportion shed > 60 s pcnt'    ,    
        13 :'proportion recruited < 15 s pcnt' ,  
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
    #p = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/regions_data/230212_regionsdata_8var.csv'
    #p = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/regions_data/230212_regionsdata_8var_trk1_minute.csv'
    p = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/regions_data/230212_regionsdata_12var_trk1_minute.csv'
    data = pd.read_csv(p)
    #regions_lineplots(data, (names[0], names[1]))
    #regions_lineplots(data, (names[3], ), time_col='minute', errorbar=True)
    #regions_lineplots(data, ('recruitment (s^-1)', ), time_col='minute', errorbar=True)
    #regions_lineplots(data, ('recruitment (s^-1)', ), time_col='minute', treatements=('DMSO (MIPS)', 'DMSO (SQ)', 'saline'), errorbar=True)
    #data = data[data['time (s)'] < 191 / 0.321764322705706 ]
    #regions_lineplots(data, (names[4], ), time_col='minute', errorbar=True)
    #regions_lineplots(data, ('shedding (s^-1)', ), time_col='minute', errorbar=True)
    #regions_lineplots(data, ('shedding (s^-1)', ), treatements=('DMSO (MIPS)', 'DMSO (SQ)', 'saline'), time_col='minute', errorbar=True)
    #regions_lineplots(data, (names[5], names[6], names[7]), time_col='minute', errorbar=True)
    #regions_lineplots(data, (names[5], names[6], names[7]), time_col='minute', errorbar=True)
    #regions_lineplots(data, (names[5], names[6], names[7]), treatements=('DMSO (MIPS)', 'DMSO (SQ)', 'saline'), time_col='minute', errorbar=True)

    #regions_lineplots(data, (names_pcnt[7], ), time_col='minute', errorbar=True) 
    #regions_lineplots(data, (names_pcnt[8], names_pcnt[9], names_pcnt[10],), time_col='minute', errorbar=True) 
    regions_lineplots(data, (names_pcnt[11], names_pcnt[12],), time_col='minute', errorbar=True) 
