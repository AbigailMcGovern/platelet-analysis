#---------------------------------------------------------------------------
# FUNCTIONS FOR DATA MANIPULATION, 
#---------------------------------------------------------------------------
#CONSIDER USING df.assign() instead!

import pandas as pd
import numpy as np
from .. import config as cfg
import sys
from pathlib import Path
from IPython.display import clear_output  
import pingouin as pg
import os
import time
import matplotlib.pyplot as plt
import math as m
from pathlib import Path
from numpy import False_
from . import menu_func as mfc


#---------------------------------------------------------------------------
# FUNCTIONS DEFINING VARIABLES, CONSIDER USING df.assign() instead!
#---------------------------------------------------------------------------

def led_bins_var(dfg):#Creates bin variables tled & zled  for heatmaps etc. 
    zbins=[-2,4]+np.arange(8,72,4).tolist()#np.arange(-8,72,4)
    zgroup_names=np.round(np.linspace(0,68,17),0)#np.round(np.linspace(-6,74,19),0)
    #tbins=np.arange(0,196,2)
    tbins=np.linspace(0,194,98)#np.linspace(0,192,97)
    tgroup_names=np.round(np.linspace(0,600,97),0)
    #tgroup_names=np.arange(0,196,2)
    if 'zs' in dfg.columns:  
        dfg['zled'] = pd.cut(dfg['zs'], zbins, labels=zgroup_names,include_lowest=True).astype('Int64')
    if 'sec' in dfg.columns:
        dfg['tled'] = pd.cut(dfg['frame'], tbins, labels=tgroup_names,include_lowest=True).astype('Int64')    
    return dfg

def dist_c_var(df):# Creates variables dist_c & dist_cz that give distance from center
    df['dist_c']=((df.loc[:,'x_s'])**2+(df.loc[:,'ys'])**2)**0.5
    df['dist_cz']=((df.loc[:,'x_s'])**2+(df.loc[:,'ys'])**2+(df.loc[:,'zs'])**2)**0.5
    return df

def isovol_bin_var(df):
    if 'dist_cz' in df.columns:
        inj_zone_vol=(2/3)*m.pi*(37.5**3)
        vol_step=inj_zone_vol/3#inj_zone_vol/10
        volumes_=np.arange(0,vol_step*61,vol_step)#np.arange(0,vol_step*201,vol_step)
        radii=((3*volumes_/(2*m.pi)))**(1/3)
        radii[-1]=250
        df['iso_vol']=pd.cut(df['dist_cz'],radii,labels=radii[1:]).astype('float64')
        #vol_step=20000
        #n_=np.arange(0,210)
        #radii=((3*vol_step*n_/(2*m.pi)))**(1/3)
        #radii[-1]=250.0
        #df['iso_vol']=pd.cut(df['dist_cz'],radii,labels=radii[1:])
    return df

def isovol_bin_var1(df):
    if 'dist_cz' in df.columns:
        inj_zone_vol=(2/3)*m.pi*(37.5**3)
        vol_step=inj_zone_vol/3
        volumes_=np.arange(0,vol_step*61,vol_step)
        radii=((3*volumes_/(2*m.pi)))**(1/3)
        radii[-1]=250
        df['iso_vol']=pd.cut(df['dist_cz'],radii,labels=radii[1:]).astype('float64')
     
    return df

def time_var(df):# Creates time variables time, minute and phase from frame
    df['sec']=df['frame']*3.1
    return df

def minute_var(df):
    if 'sec' not in df.columns:
        df=time_var(df)
    df.loc[:,'minute'] = pd.cut(df['sec'], 10, labels=np.arange(1,11,1))
    return df

def phase_var(df):
    df.loc[:,'phase'] = pd.qcut(df['frame'], 3, labels=['Early','Mid','Late'])
    return df

def injury_zone_var(df):# Creates variable that divide plts into thos inside and outside injury zone
    if 'dist_cz' in df.columns.tolist():
        df['injury_zone']=df.dist_cz<38
    else:
        df=dist_c_var(df)
        df['injury_zone']=df.dist_cz<38
    return df

def height_var(df):
    df.loc[:,'height']=pd.cut(df.zf, 3, labels=["bottom", "middle", "top"])
    return df

def z_pos_var(df):
    df.loc[:,'z_pos']=pd.cut(df.zf, 2, labels=["bottom", "top"])
    return df

def zz_var(df):
    df['zz']=np.round(df.loc[:,'zf'],decimals=0)#pc['zz']=np.round(pc.loc[:,'zs'],decimals=0)
    df = df.astype({'zz': int})
    return df

def position_var(df):
    df['position']='outside'
    df.loc[(df.dist_c<38),'position']='head'
    df.loc[(df.dist_c>38)&(df.ys<0)&(abs(df.x_s)<38),'position']='tail'
    return df

def quadrant_var(df):
    df['quadrant']='lateral'
    df.loc[(df['ys']>df['x_s'])&(df['ys']>-df['x_s']),'quadrant']='anterior'
    df.loc[(df['ys']<df['x_s'])&(df['ys']<-df['x_s']),'quadrant']='posterior'
    return df

def quadrant1_var(df):
    if 'injury_zone' not in df.columns:
        df=injury_zone_var(df)
    df['quadrant1']='lateral'
    df.loc[(df['ys']>df['x_s'])&(df['ys']>-df['x_s']),'quadrant1']='anterior'
    df.loc[(df['ys']<df['x_s'])&(df['ys']<-df['x_s']),'quadrant1']='posterior'
    df.loc[df.injury_zone,'quadrant1']='core'
    return df

def inside_injury_var(df): 
    df['inside_injury']=df.position.isin(['head'])
    return df

def mov_class_var(df):#New definition 191209
    for exp_id in pd.unique(df.exp_id):
        dfi=df[df.exp_id==exp_id].copy()
    try:
        still=pd.unique(dfi[((dfi.displ_tot/dfi.nrtracks)<0.1)&(dfi.displ_tot<4)].particle)
        loose=pd.unique(dfi[(dfi.displ_tot>5)&((dfi.cont_tot/dfi.displ_tot)<0.2)].particle)
        contractile=pd.unique(dfi[((dfi.cont_tot/dfi.displ_tot)>0.5)&(dfi.displ_tot>1)].particle)
    except TypeError:
        print(exp_id,dfi.displ_tot.dtypes,dfi.nrtracks.dtypes,(dfi.displ_tot/dfi.nrtracks))
    df.loc[(df.exp_id==exp_id)&(df['particle'].isin(still)),'mov_class']="still"
    df.loc[(df.exp_id==exp_id)&(df['particle'].isin(loose)),'mov_class']="loose"
    df.loc[(df.exp_id==exp_id)&(df['particle'].isin(contractile)),'mov_class']="contractile"
    return df

def movement_var(df):
    df['movement']='none'
    df.loc[(df.dv<0.1) & (df.tracked),'movement']='immobile' #pc.loc[(pc.dv)<0.3,'movement']='still'
    df.loc[(abs(df.dvy/df.dv)>0.5) & (df.dvy<0),'movement']='drifting' #pc.loc[(pc.dv>0.3)&(pc.cont_p<0.5),'movement']='drifting'
    df.loc[(df.dv>0.3) & (df.cont_p>0.5) ,'movement']='contracting'
    df.loc[(df.stab>3),'movement']='unstable'
    return df

def scale_var(df,var1):
    var2=var1+'_s'
    df[var2]=df[var1]*1000/3.1
    print(var2)
    return df

def binning_labels_var(dfg,binned_var,bins):
    #bin_labels=
    dfg[binned_var]=pd.cut(dfg[binned_var],bins,precision=0)
    bin_var=f'{binned_var}_binlabel'
    bin_labels=[]
    for bin in dfg[binned_var].sort_values().unique():
        bin_label=str(np.round(np.mean((bin.right+bin.left)/2),0))
        bin_labels.append(bin_label)
        #print (bin,bin_label)
        dfg.loc[dfg[binned_var]==bin,bin_var]=bin_label
    bin_order = sorted(bin_labels, key=lambda x: float(x))
    return dfg,bin_var,bin_order

def qbinning_labels_var(dfg,binned_var,bins):
    dfg[binned_var]=pd.qcut(dfg[binned_var],bins,precision=0)
    bin_var=f'{binned_var}_binlabel'
    bin_labels=[]
    for bin in dfg[binned_var].sort_values().unique():
        bin_label=str(int(np.round(np.mean((bin.right+bin.left)/2),0)))
        bin_labels.append(bin_label)
        #print (bin,bin_label)
        dfg.loc[dfg[binned_var]==bin,bin_var]=bin_label
    bin_order = sorted(bin_labels, key=lambda x: float(x))
    return dfg,bin_var,bin_order

def qbinning_quant(dfg,binned_var,bins):
    quint=np.arange(1,bins+1,1)
    labels=[]
    for n in quint:
        labels.append(str(n*10))#labels.append(str(n)+'th')
    dfg[binned_var]=pd.qcut(dfg[binned_var],bins,labels=labels,precision=0)
    return dfg,binned_var,labels

def new_exp_ids(pc):
    pc["exp_id"] = pc.groupby(["path"]).ngroup()
    for inh in pc.inh.unique():
        pc.loc[pc.inh==inh,'inh_exp_id']=pc[pc.inh==inh].groupby('path').ngroup()#grouper.group_info[0]
    return pc

def inh_names_2lines(df):
    longnames_list=['Bivalirudin','Cangrelor','CMFDA','Control',
                 'MIPS','Saline','SQ','Vehicle MIPS',
                 'Vehicle SQ','PAR4+/-','PAR4-/+','PAR4-/- + biva',
                 'PAR4-/-','ASA + Vehicle','ASA','Salgav + Vehicle',
                 'Salgav'
                ]
    inh_dic={'vehicle MIPS':'vehicle\nMIPS', 'salgavDMSO':'salgav\nDMSO', 'vehicle sq':'vehicle\nsq', 'par4--biva':'par4--\nbiva'}

#---------------------------------------------------------------------------
# FUNCTIONS FOR MANIPULATING OTHER DATASTRUCTURES
#---------------------------------------------------------------------------
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

#Function for selecting unique elements in list while preserving order
#-------------------------------------------------------------
def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

#---------------------------------------------------------------------------
# BUILD DATA STRUCTURES WITH/WITHOUT MENUS 
#---------------------------------------------------------------------------
# Build dataframes for analysis
#---------------------------------------------------------------------------
def build_df_lists(col_list,treatments):#Builds dataframe from lists of variables and list (inh_order) with treatments
    global inh_order,inh_list
    inh_order=treatments
    inh_list=[cfg.longtoshort_dic[inh] for inh in inh_order]
    df_=[]
    paths = Path(cfg.df_paths.path[0]).glob('**/*')
    paths = [x for x in paths if x.is_file()]
    list1=[path for inh in inh_list for path in paths if inh in path.name]
    #print(inh_list)
    path_list=list(set(list1))
       
    for n_df,fi in enumerate(path_list):
        dfi=pd.read_parquet(fi)#dfi=pd.read_pickle(fi)
        dfi=dfi[dfi.frame<194]# Remove rows with data from frames > 193
        dfi_col=[]#'path', 'inh','particle'
        absent_cols=[]
        for col in col_list:
            if col in dfi.columns:
                dfi_col.append(col)
            else:
                absent_cols.append(col)      
        df_.append(dfi.loc[:,dfi_col])
        if absent_cols:
            print(f'columns absent in {fi}:{absent_cols}')
    pc=pd.concat(df_, ignore_index=True)#.reset_index()#,names='pid'
    
    for inh in pc.inh.unique():
        pc.loc[pc.inh==inh,'inh']=cfg.shorttolong_dic[inh]#.casefold()
    if 'minute' in pc.columns:
        pc.loc[:,'minute'] = pd.cut(pc['time'], 10, labels=np.arange(1,11,1))
    
    pc=new_exp_ids(pc)
    #pc=pc.drop(['level_0','index'],axis=1).reset_index()
    #print(pc.columns)
    #if 'level_0' in pc.columns:
    #    pc=pc.drop(['level_0'],axis=1)
    #if 'index' in pc.columns:
    #    pc=pc.drop(['index'],axis=1)
   # pc=pc.reset_index()
    pc.index.name = 'pid'
    pc=pc.reset_index()
    pc=pc.rename(columns={'time':'sec'})
  #  #clear_output(wait=False)
    print(f'Treatments = {pc.inh.unique()}',flush=True) #Paths included = {pc.path.unique()}\n
    print(f'RESULTING DATAFRAME\nNo of columns: {pc.shape[1]} \nNo of rows: {pc.shape[0]}',
          f'\nMemory use: {np.round(sys.getsizeof(pc)/1000000,1)} Mb',flush=True)
    return pc


def reorder_inh_menu():#Function for reordering treatments in list
    global inh_list,inh_order
    names1=[]
    nr_treatments=len(inh_list)
    names=[cfg.shorttolong_dic[inh] for inh in inh_list]
    for nr in range(nr_treatments):
        for n,name in enumerate(names):
            print(n,name)
        print('  ',flush=True)
        choice = int(input('Pick the treatment that will be next in order'))
        names1.append(names[choice])
        names.remove(names[choice])
    inh_order=names1
    inh_list=[cfg.longtoshort_dic[inh] for inh in names1]     


    
def show_expseries():#Function showing the available series for analysis
    print(f'Experimental cohorts available for selection:\n{73 * "-"}\n')
    n=0
    for serie_name,serie in zip(cfg.expseries_listnames,cfg.treatments_):
        #print(f'{n}:{serie_name}\t')
        list_names=[]
        for inh in serie:
            list_names.append(cfg.shorttolong_dic[inh])
        print(f'{n}:{serie_name}\n{list_names}')
        print('\n',flush=True)
        n+=1



def df_outliers_menu(df_outliers,df_inliers):
        df_outliers=df_outliers.loc[:,['measure','inh','path','value']]
        df_inliers=df_inliers.reset_index().loc[:,['measure','inh','path','value']]
        print(df_outliers.to_markdown(),'\n')
        col_vars=input(f'Choose which experiments you want to save as outliers \nEnter your choice as integers separated by spaces\n{73 * "-"}\n')
        flist = [int(x) for x in col_vars.split()]
        df_choice=df_outliers.iloc[flist]
        
        choice = input('Do you want to add additional experiments in the outliers file? (y/n)')
        if choice =='y':
            print(df_inliers.to_markdown(),'\n')
            print('Choose which experiments you want to save as outliers',flush=True)
            col_vars=input(f'Enter your choice as integers separated by spaces\n{73 * "-"}\n')
            flist = [int(x) for x in col_vars.split()]
            df_choice1=df_inliers.iloc[flist]
            print(df_choice1.to_markdown())
            df_choice=pd.concat([df_choice,df_choice1])
        #print(df_choice.to_markdown())
        choice = input('Do you want to save your results in the outliers file? (y/n)')
        if choice =='y':
            df_choice.to_csv('df_outliers.csv')

def xtravars_menu():
    print('One of the following variables can be used to analyse different regions of the thrombus separately:')
    for c, value in enumerate(cfg.thr_reg_vars,0):
        print(c, value) 
    print(' ',flush=True)
    var_nr=int(input("Enter your choice:"))
    var=cfg.thr_reg_vars[var_nr]
    return var

def add_xtravars(df,xtra_vars):
    if isinstance(xtra_vars, dict):
        xtra_var_ls=xtra_vars.values()
    if 'phase' in xtra_var_ls:
        df=phase_var(df)
    if 'injury_zone' in xtra_var_ls:
        df=injury_zone_var(df)
    if 'minute' in xtra_var_ls:
        df=minute_var(df)
    if 'position' in xtra_var_ls:
        df=position_var(df)
    if 'height' in xtra_var_ls:
        df=height_var(df)
    if 'z_pos' in xtra_var_ls:
        df=z_pos_var(df)
    if 'inside_injury' in xtra_var_ls:
        df=inside_injury_var(df)
    return df

def add_xtravar(df,xtra_var):
    #if xtra_var == 'inside_injury':
    #    df=inside_injury_var(df)
    if xtra_var == 'injury_zone':
        df=injury_zone_var(df)
    elif xtra_var == 'minute':
        df=minute_var(df)
    #elif xtra_var == 'position':
    #    df=position_var(df)
    elif xtra_var == 'height':
        df=height_var(df)
    elif xtra_var == 'z_pos':
        df=z_pos_var(df)
    elif xtra_var == 'phase':
        df=phase_var(df)
    elif xtra_var == 'quadrant':
        df=quadrant_var(df)
    elif xtra_var == 'quadrant1':
        print('quadrant1 added')
        df=quadrant1_var(df)
    return df

# Create grouped dataframes (e.g. for plotting)
#---------------------------------------------------------------------------
def rolling_count(df,grouping_var,x_var): ##Grouped df with rolling counts, difference in 'diff' column 
    dv_=[]
    dg_=[]
    grouping_vars=grouping_var+['inh','path']#'inh_exp_id'
    grouping1=grouping_vars+x_var
    #if x_var!=['sec']:
    #    grouping1=grouping1+['sec']
    df_grouped=df.groupby(grouping1).count()[['pid']]#.rename(columns={'pid':'plts'})#
    df_grouped=df_grouped.reset_index()
    dfg=df_grouped.groupby(grouping_vars)
    for i,gr in dfg:
    
        df2=gr[['pid']].rolling(window=6,win_type='bartlett',min_periods=3,center=True).mean()#.\
        dv_.append(df2)
        df_gr=df2.diff()
        dg_.append(df_gr)
    dv=pd.concat(dv_, axis=0)
    df_grouped['roll']=dv*40
    d_gr=pd.concat(dg_, axis=0)
    df_grouped['diff']=d_gr*40
    if x_var==['minute']:
        grouping2=grouping_vars+x_var
        df_grouped=df_grouped.groupby(grouping2).mean()[['roll']].reset_index()
    return df_grouped

def rolling_mean(df,grouping_var,y_var,x_var): #Rolling mean values for variable y_var 
    dv_=[]
    #x_var=['sec']
    if grouping_var=='inh':
        grouping_vars=[grouping_var]+['inh_exp_id']
    else:
        grouping_vars=[grouping_var]+['inh','inh_exp_id']
    grouping1=grouping_vars+x_var
    if x_var!=['sec']:
        grouping1=grouping1+['sec']
        print(grouping1)
    # First grouping with time
    df_grouped=df.groupby(grouping1).mean()[[y_var]]#.rename(columns={'pid':'plts'})#
    df_grouped=df_grouped.reset_index()
    #Second grouping without time 
    dfg=df_grouped.groupby(grouping_vars)
    #Rolling
    for i,gr in dfg:
    
        df2=gr[[y_var]].rolling(window=8,win_type='blackman',min_periods=3,center=True).mean()#.\'bartlett'
        dv_.append(df2)
    dv=pd.concat(dv_, axis=0)
    df_grouped['roll']=dv#*40
    return df_grouped

def rolling_perc(df1,df2,grouping_var,x_var): #Rolling percentiles for df2/df1
    dv_=[]
    dg_=[]
    grouping_vars=grouping_var+['inh','inh_exp_id']
    grouping1=grouping_vars+x_var
    if x_var!=['sec']:
        grouping1=grouping1+['sec']
    dfg1=df1.groupby(grouping1).count()['pid']#.reset_index()
    dfg2=df2.groupby(grouping1).count()['pid']
    dfg=dfg2/dfg1
    df_grouped=dfg.reset_index() 
    dfg=df_grouped.groupby(grouping_vars)
    for i,gr in dfg:
        df2=gr[['pid']].rolling(window=8,win_type='blackman',min_periods=3,center=True).mean()
        dv_.append(df2)
        df_gr=df2.diff()
        dg_.append(df_gr)
    dv=pd.concat(dv_, axis=0)
    df_grouped['roll']=dv*100
    d_gr=pd.concat(dg_, axis=0)
    df_grouped['diff']=d_gr*100
    if x_var==['minute']:
        grouping2=grouping_vars+x_var
        df_grouped=df_grouped.groupby(grouping2).mean()[['roll']].reset_index()
    return df_grouped

def rolling_bartlett(gr,y_var):
    gr[f'{y_var}_roll']=gr[y_var].rolling(window=6,win_type='bartlett',min_periods=3,center=True).mean()
    gr['count_roll']=gr['count'].rolling(window=4,win_type='bartlett',min_periods=3,center=True).mean()
    return gr

def rolling_mean_zled(df,y_var):
    dfg=df.groupby(['inside_injury','inh','zled','sec']).mean().reset_index()#Lägg till inh_exp_id senare!
    df_count=df.groupby(['inside_injury','inh','zled','sec']).count()['pid'].reset_index()#Lägg till inh_exp_id senare!
    dfg['count']=df_count['pid']
    dfg=dfg.groupby(['inside_injury','inh','zled']).filter(lambda x:x['count'].sum() >1000)
    dfg=dfg.groupby(['inside_injury','inh','zled']).apply(rolling_bartlett)
    dfg=dfg[['inside_injury','inh','zled','sec',y_var,f'{y_var}_roll','count']]
    dfg=dfg.reset_index()
    return dfg

def rolling_timecount(df,grouping_var,x_var):
    #grouping_var=[y_var]
    #x_var=[x_var]
    df_grouped=rolling_count(df,grouping_var,x_var)
    return df_grouped

# Statistics
#---------------------------------------------------------------------------
def build_df_statcounts(inh_order):

    time=['frame','time']
    hue_vars=[]
    new_vars=[]
    #var1,var2=thrombus_parts_menu()
    xtra_vars=huevars_stat_menu()
    for var in xtra_vars.values():
        if var in cfg.old_huevars_stat:
            hue_vars.append(var)
        else:
            new_vars.append(var)

    df_var_list=[time,hue_vars]  
    df=build_df_lists(df_var_list,inh_order)
    df=add_xtravars(df,new_vars)
    
    return df,xtra_vars

def stats_df(df_test,test_var):
    nrrows=len(inh_order)**2
    index = pd.MultiIndex.from_product([inh_order,inh_order])
    d = pd.DataFrame({'MWU': np.ones(nrrows),'ttest': np.ones(nrrows)},index=index)
    for i in (inh_order):
        for j in (inh_order):
            try:
                p=pg.mwu(df_test.loc[df_test['inh']==i,test_var],df_test.loc[df_test['inh']==j,test_var], alternative='less')
                d.loc[(i,j)]['MWU']=p['p-val']
                p=pg.ttest(df_test.loc[df_test['inh']==i,test_var],df_test.loc[df_test['inh']==j,test_var], alternative='less')
                d.loc[(i,j)]['ttest']=p['p-val']
            except TypeError:
                    print('TypeError in',test_var)
    return(d)

def huevars_stat_menu():
    print('Apart from total plt counts, the following variables can also be included in statistical comparisons:')
    for c, value in enumerate(cfg.thr_huevars_stat,0):
        print(c, value) 
    col_vars=input("Enter your choice as 0-2 integers separated by spaces:")
    varlist = [int(x) for x in col_vars.split()]
    xtra_vars={}
    for nr in range(1,len(varlist)+1):
        xtra_vars.update({'var'+str(nr):cfg.thr_huevars_stat[varlist[nr-1]]})
        #print('Original values of xtra_vars:')
        #for key,value in xtra_vars.items():
         #   print(key,value)
    return xtra_vars#var1,var2

def xtravars_stat(thr_reg_var,time_var):
    xtra_vars={}
    xtra_vars.update({'var'+str(1):thr_reg_var})
    xtra_vars.update({'var'+str(2):time_var})
        #print('Original values of xtra_vars:')
        #for key,value in xtra_vars.items():
         #   print(key,value)
    return xtra_vars#var1,var2


def stats_counts(df,xtra_vars):
    ls_desc=[]
    ls_tests=[]
    dfg_auc=df.groupby(['inh','path'])[['pid']].count().rename(columns={'pid':'count'}).reset_index()    
    df_desc=dfg_auc.groupby(['inh'])[['count']].describe() 
    df_tests_auc=stats_df(dfg_auc,'count')
    xtra_vars.update({'value1':'All','value2':'All',
                     'ls_desc':ls_desc, 'ls_tests':ls_tests})
    xtra_vars=insertvars_statdf(df_desc,df_tests_auc,xtra_vars)
    if 'var1' in xtra_vars:
        values1=df[xtra_vars['var1']].unique().tolist()
        #print(values1)
        for value1 in values1: 
            #print(value1)
            xtra_vars.update({'value1':value1})
            dfg_auc=df[df[xtra_vars['var1']]==value1].groupby(['inh','path'])[['pid']].count().rename(columns={'pid':'count'}).reset_index()
            df_desc=dfg_auc.groupby(['inh'])[['count']].describe()
            df_tests_auc=stats_df(dfg_auc,'count')
            xtra_vars=insertvars_statdf(df_desc,df_tests_auc,xtra_vars) 
    if 'var2' in xtra_vars:
        values2=df[xtra_vars['var2']].unique().tolist()
        #print(values2)
        xtra_vars.update({'value1':'All'})
        for value2 in values2:
            #print(value2)
            xtra_vars.update({'value2':value2})
            dfg_auc=df[df[xtra_vars['var2']]==value2].groupby(['inh','path'])[['pid']].count().rename(columns={'pid':'count'}).reset_index()
            df_desc=dfg_auc.groupby(['inh'])[['count']].describe()
            df_tests_auc=stats_df(dfg_auc,'count')
            xtra_vars=insertvars_statdf(df_desc,df_tests_auc,xtra_vars)
        for value1 in values1:
            xtra_vars.update({'value1':value1})
            for value2 in values2:
                xtra_vars.update({'value2':value2})
                dfg_auc=df[(df[xtra_vars['var1']]==value1)&(df[xtra_vars['var2']]==value2)].groupby(['inh','path'])[['pid']].count().rename(columns={'pid':'count'}).reset_index()
                df_desc=dfg_auc.groupby(['inh'])[['count']].describe()
                df_tests_auc=stats_df(dfg_auc,'count')
                xtra_vars=insertvars_statdf(df_desc,df_tests_auc,xtra_vars)
    df_desc=pd.concat(xtra_vars['ls_desc'],axis=0)#keys=groups,
    df_tests=pd.concat(xtra_vars['ls_tests'],axis=0)#keys=groups,
    #print(df_desc,df_tests)
            
    return df_desc,df_tests



def insertvars_statdf(df_desc,df_tests_auc,xtra_vars):
    if 'var1' in xtra_vars:
        df_desc.insert(0, xtra_vars['var1'],xtra_vars['value1'])
        df_tests_auc.insert(0,xtra_vars['var1'],xtra_vars['value1']) 
        if 'var2' in xtra_vars:
            df_desc.insert(1, xtra_vars['var2'],xtra_vars['value2'])
            df_tests_auc.insert(1,xtra_vars['var2'],xtra_vars['value2'])
    xtra_vars['ls_desc'].append(df_desc)
    xtra_vars['ls_tests'].append(df_tests_auc)
    #xtra_vars.update({'ls_desc':ls_desc, 'ls_tests':ls_tests})
    return xtra_vars

#---------------------------------------------------------------------------
#FUNCTIONS FOR STORAGE OF DATA & PLOTS
#---------------------------------------------------------------------------

        
def makedir(results_folder):
    try:
        os.mkdir(results_folder)
    except FileExistsError: 
        print(f'Folder {results_folder} already exists')


def save_fig(test_var,*xtra):#,formats,**xtra1
    if mfc.save_figs:
        file_name=time.strftime("%Y%m%d") +'_'+ test_var
        if mfc.save_inh_names:
            inh_shortlist=[cfg.longtoshort_dic[inh] for inh in inh_order]
            inhibitors=''.join(inh_shortlist)
            file_name=file_name+'_'+inhibitors    
        for stuff in xtra:
            file_name=file_name+'_'+stuff
        plot_path_png=f'{mfc.results_folder}\\{file_name}'
        plot_path_svg=plot_path_png
        if mfc.plot_formats=='b':
            plt.savefig(plot_path_png+'.png',bbox_inches='tight', dpi=300)
            plt.savefig(plot_path_svg+'.svg',bbox_inches='tight')
        if mfc.plot_formats=='s':
            plt.savefig(plot_path_svg+'.svg',bbox_inches='tight')
        else:
            plt.savefig(plot_path_png+'.png',bbox_inches='tight', dpi=300)

def save_table(df,*xtra):#,formats,**xtra1
    if mfc.save_figs:   
        file_name=time.strftime("%Y%m%d") #+'_'+ test_var
        if mfc.save_inh_names==True:
            inhibitors=''.join(inh_order)
            file_name=file_name+'_'+inhibitors
        for stuff in xtra:
            file_name=file_name+'_'+stuff
        plot_path_csv=f'{mfc.results_folder}\\{file_name}'
        df.to_csv(plot_path_csv+'.csv')


