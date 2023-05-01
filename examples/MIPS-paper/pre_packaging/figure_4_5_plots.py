import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


def regions_lineplots(
        data, 
        variables,
        different_treatements=False,
        treatements=('MIPS', 'SQ', 'cangrelor'), 
        regions=('center', 'anterior', 'lateral', 'posterior'), 
        time_col='time (s)',
        hue='treatment', 
        errorbar=False,
        log=False
    ):
    sns.set_context('paper')
    sns.set_style('ticks')
    #if not log and len(variables) > 1:
    #    log = [False, ] * len(variables) 
    #if log and len(variables) > 1:
     #   log = [True, ] * len(variables) 
    if different_treatements:
        assert len(treatements) == len(variables)
    if not different_treatements:
        dfs = []
        for t in treatements:
            sdf = data[data['treatment'] == t]
            dfs.append(sdf)
        data = pd.concat(dfs).reset_index(drop=True)
    else:
        data_list = []
        for group in treatements:
            dfs = []
            for t in group:
                sdf = data[data['treatment'] == t]
                dfs.append(sdf)
            gdf = pd.concat(dfs).reset_index(drop=True)
            data_list.append(gdf)
        data = data_list
    del dfs
    if errorbar:
        es = 'bars'
        m = 'o'
    else:
        es = 'band'
        m = None
    matplotlib.rcParams.update({'font.size': 6})
    fig, axs = plt.subplots(len(regions), len(variables), sharex='col')
    plt.xticks(rotation=45)
    for j in range(len(regions)):
        r = regions[j]
        if not different_treatements:
            sdf = data[data['region'] == r]
        if len(variables) > 1:
            for i in range(len(variables)):
                if different_treatements:
                    sdf = data[i][data[i]['region'] == r]
                ax = axs[j, i]
                ax.set_title(r)
                sns.despine(ax=ax)
                #if log[i]:
                 #   ax.set_yscale('log')
                sns.lineplot(data=sdf, x=time_col, y=variables[i], hue=hue, ax=ax, errorbar=("se", 1), err_style=es, marker=m)
                if 'pcnt' in variables[i]:
                    ax.set_ylabel('percent vehicle (%)')
                    ax.axline((0, 100), (1, 100), color='grey', alpha=0.5)
                if time_col == 'hsec':
                    ax.set_xlabel('time post injury (s)')
                    for label in ax.get_xticklabels():
                        label.set_rotation(45)
                        label.set_ha('right')
                if time_col == 'time (s)':
                    ax.set_xlabel('time post injury (s)')
                if time_col == 'minute':
                    ax.set_xlabel('time post injury (min)')
        else:
            ax = axs[j]
            ax.set_title(r)
            sns.despine(ax=ax)
            #if log:
             #   ax.set_yscale('log')
            sns.lineplot(data=sdf, x=time_col, y=variables[0], hue=hue, ax=ax, errorbar=("se", 1), err_style=es, marker=m)
    fig.subplots_adjust(right=0.95, left=0.17, bottom=0.11, top=0.95, wspace=0.45, hspace=0.4)
    fig.set_size_inches(4.5, 7)
    #plt.xticks(rotation=45)
    plt.show()


#def regions_barcharts()


if __name__ == '__main__':
    names=('platelet count pcnt', 'platelet density (um^-3) pcnt', 'thrombus edge distance (um) pcnt', 
               'recruitment (s^-1) pcnt', 'shedding (s^-1) pcnt', 'mean stability pcnt', 
               'dvy (um s^-1) pcnt', 'mean tracking time (s) pcnt')
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
    #p = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/regions_data/230212_regionsdata_8var.csv'
    #p = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/regions_data/230212_regionsdata_8var_trk1_minute.csv'
    #p = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/regions_data/230212_regionsdata_12var_trk1_minute.csv'
    #p = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/regions_data/230305_regionsdata_12var_trk1_minute.csv'
    p = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/regions_data/230405_regionsdata_9var_trk1_seconds.csv'
    #p = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/regions_data/230405_regionsdata_12var_trk1_minute.csv'
    #p = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/regions_data/230405_regionsdata_12var_trk1_hsec.csv'
    #230305_regionsdata_9var_trk1_seconds

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
    #regions_lineplots(data, (names[7], ), time_col='minute', errorbar=True, treatements=('MIPS', 'DMSO (MIPS)')) 
    #regions_lineplots(data, (names[8], names[9], names[10],), time_col='minute', errorbar=True, treatements=('MIPS', 'DMSO (MIPS)')) 
    #regions_lineplots(data, (names[11], names[12],), time_col='minute', errorbar=True, treatements=('MIPS', 'DMSO (MIPS)')) 
    #regions_lineplots(data, (names_pcnt[11], names_pcnt[12],), time_col='minute', errorbar=True,) # log=(True, False)) 

    #regions_lineplots(data, (names[0], names[1]), time_col='minute', errorbar=True, treatements=('MIPS', 'DMSO (MIPS)')) 
    #regions_lineplots(data, (names_pcnt[0], names_pcnt[1]), time_col='minute', errorbar=True) 
    #regions_lineplots(data, (names[11], names_pcnt[11],), time_col='minute', 
     #                 errorbar=True, treatements=(('MIPS', 'DMSO (MIPS)'), ('MIPS', 'SQ', 'cangrelor')), different_treatements=True) 
    
    #regions_lineplots(data, (names[13], names_pcnt[13],), time_col='minute', 
     #                 errorbar=True, treatements=(('MIPS', 'DMSO (MIPS)'), ('MIPS', 'SQ', 'cangrelor')), different_treatements=True) 

    regions_lineplots(data, (names[0], names_pcnt[0],), time_col='time (s)', 
                      errorbar=False, treatements=(('MIPS', 'DMSO (MIPS)'), ('MIPS', 'SQ', 'cangrelor')), 
                      different_treatements=True) 
    
    #regions_lineplots(data, (names[1], names_pcnt[1],), time_col='minute', 
    #                  errorbar=True, treatements=(('MIPS', 'DMSO (MIPS)'), ('MIPS', 'SQ', 'cangrelor')), 
    #                  different_treatements=True) 
    
    #regions_lineplots(data, (names[3], names_pcnt[3],), time_col='minute', 
    #                  errorbar=True, treatements=(('MIPS', 'DMSO (MIPS)'), ('MIPS', 'SQ', 'cangrelor')), 
    #                  different_treatements=True) 

    #regions_lineplots(data, (names[4], names_pcnt[4],), time_col='minute', 
    #                  errorbar=True, treatements=(('MIPS', 'DMSO (MIPS)'), ('MIPS', 'SQ', 'cangrelor')), 
    #                  different_treatements=True) 
    
    #regions_lineplots(data, (names[5], names_pcnt[5],), time_col='minute', 
    #                  errorbar=True, treatements=(('MIPS', 'DMSO (MIPS)'), ('MIPS', 'SQ', 'cangrelor')), 
    #                  different_treatements=True) 

    #regions_lineplots(data, (names[6], names_pcnt[6],), time_col='minute', 
    #                  errorbar=True, treatements=(('MIPS', 'DMSO (MIPS)'), ('MIPS', 'SQ', 'cangrelor')), 
    #                  different_treatements=True) 
    
    #regions_lineplots(data, (names[7], names_pcnt[7],), time_col='minute', 
    #                  errorbar=True, treatements=(('MIPS', 'DMSO (MIPS)'), ('MIPS', 'SQ', 'cangrelor')), 
    #                  different_treatements=True) 
    
    #regions_lineplots(data, (names[8], names_pcnt[8],), time_col='minute', 
    #                  errorbar=True, treatements=(('MIPS', 'DMSO (MIPS)'), ('MIPS', 'SQ', 'cangrelor')), 
    #                  different_treatements=True) 

    #regions_lineplots(data, (names[13], names_pcnt[13],), time_col='minute', 
    #                   errorbar=True, treatements=(('MIPS', 'DMSO (MIPS)'), ('MIPS', 'SQ', 'cangrelor')), 
    #                  different_treatements=True) 
    
    #regions_lineplots(data, (names[14], names_pcnt[14],), time_col='minute', 
    #                   errorbar=True, treatements=(('MIPS', 'DMSO (MIPS)'), ('MIPS', 'SQ', 'cangrelor')), 
    #                  different_treatements=True) 
    
    #regions_lineplots(data, (names[1], names_pcnt[1],), time_col='hsec', 
    #                   errorbar=True, treatements=(('MIPS', 'DMSO (MIPS)'), ('MIPS', 'SQ', 'cangrelor')), 
    #                  different_treatements=True)
    
    #regions_lineplots(data, (names[3], names_pcnt[3],), time_col='hsec', 
    #                   errorbar=True, treatements=(('MIPS', 'DMSO (MIPS)'), ('MIPS', 'SQ', 'cangrelor')), 
    #                  different_treatements=True)
    
    #regions_lineplots(data, (names[13], names_pcnt[13],), time_col='hsec', 
    #                   errorbar=True, treatements=(('MIPS', 'DMSO (MIPS)'), ('MIPS', 'SQ', 'cangrelor')), 
    #                  different_treatements=True)
    
    #print(data['treatment'].value_counts())
    #data['recruitment (min^-1)'] = data[names[3]] * 0.321764322705706
    #regions_lineplots(data, ('recruitment (min^-1)', names_pcnt[3],), time_col='minute', 
     #                 errorbar=True, treatements=(('MIPS', 'DMSO (MIPS)'), ('MIPS', 'SQ', 'cangrelor')), different_treatements=True) 

    #p = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/regions_data/230215_regionsdata_9var_trk1_seconds.csv'
    #data = pd.read_csv(p)
    #regions_lineplots(data, (names[0], names_pcnt[0],), treatements=(('MIPS', 'DMSO (MIPS)'), ('MIPS', 'SQ', 'cangrelor')), different_treatements=True) 

