

#---------------------------------------------------------------------------
# PLOTTING FUNCTIONS
#---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec
import numpy as np
import os
import statsmodels.api as sm
import math as m
from IPython.display import clear_output   
import pingouin as pg
#import tabulate
from scipy import ndimage
from scipy import stats
from . import data_func_magic as dfc
from .. import config as cfg
from importlib import reload
reload(cfg)

leg_titles=dict(inh='Treatment',position='Position', inside_injury='Inside Injury XY', 
injury_zone='Inside Injury Zone',path='Experiment',movement='Movement')

def params_choice(choice='line'):
    #---------------------------------------------------------------------------
    # Creates dictionaries to send as kwargs to ploting functions
    #---------------------------------------------------------------------------
    #global pms
    if choice == 'line':
        pms=dict(x='sec',y='roll',hue='inh',ci=70, kind="line", hue_order=dfc.inh_order,
                     height=4,aspect=1.25)
    elif choice == 'line_indexp':
        pms=dict(x='sec', y='roll',hue='path',kind='line',row='inh', row_order=dfc.inh_order, 
                    height=4, aspect=1.4,legend='full')
    elif choice == 'point':
        pms=dict(x="minute",hue='inh',hue_order=dfc.inh_order, ci=70, kind="point", height=6,aspect=1.25)
    elif choice == 'heat':
        pms=dict(orient='horizontal',groups=dfc.inh_order,var='ca_corr', smooth='gauss',count_thr=0, 
                xticklabels=16,yticklabels=14, cmap="viridis",vmax=100,vmin=0)#yticklabels=2
    return pms


def create_allparams():
    #---------------------------------------------------------------------------
    # Creates dictionaries to send as kwargs to ploting functions
    #---------------------------------------------------------------------------
    #global pms_line,pms_indexp,pms_point,pms_heat,pms_dic
    pms_line=dict(x='sec',hue='inh',ci=70, kind="line", hue_order=dfc.inh_order,
                     height=4,aspect=1.25,legend=True)
    pms_indexp=dict(x='sec', y='roll',hue='path',col='inside_injury', kind='line',row='inh', row_order=dfc.inh_order, 
                    height=4, aspect=1.4,legend='full')
    pms_point=dict(x="minute",hue='inh',hue_order=dfc.inh_order, ci=70, kind="point", height=6,aspect=1.25)
    pms_heat=dict(orient='horizontal',groups=dfc.inh_order,var='ca_corr', smooth='gauss',count_thr=0, 
                xticklabels=16,yticklabels=14, cmap="turbo",vmax=100,vmin=0)#yticklabels=2
    pms_dic=dict(line=pms_line,line_indexp=pms_indexp,point=pms_point,heat=pms_heat)

    return pms_dic

#------------------------------------------------------
#LINEPLOTS COUNTS/FRACTIONS
#------------------------------------------------------
#,grouping_var='inside_injury',x_var='sec'
def lineplot_count_indexp(df,col='inside_injury'):# LINEPLOT WITH ROLLING PLT COUNT OVER TIME, INDIVIDUAL EXPERIMENTS
    params=params_choice('line_indexp')
    x_var=params['x']
    x_ls=[x_var]
    grouping_var_ls=[col]
    dfg=dfc.rolling_timecount(df,grouping_var_ls,x_ls)
    params.update({'data':dfg,'col':col,'col_order':cfg.var_order[col]})
    with sns.plotting_context("notebook", rc={"lines.linewidth": 1}):
        g=sns.relplot(**params)
        g.set_ylabels('Platelet count')
        g.set_xlabels(f'Time ({x_var})')
        g._legend.set_title(leg_titles[params['hue']])
        dfc.save_fig('count ind exp',f'lineplot {col}')
        plt.show()

#------------------------------------------------------
def lineplot_count_all(df):# LINEPLOT WITH ROLLING PLT COUNT OVER TIME, ALL PLATELETS
    grouping_var=[]
    params=params_choice('line')
    x_var=params['x']
    x_ls=[x_var]
    dfg=dfc.rolling_timecount(df,grouping_var,x_ls)
    params.update({'data':dfg})
    g=sns.relplot(**params)#legend=False
    g.fig.set_size_inches(12,6)
    g.set_ylabels('Platelet count')
    g.set_xlabels(f'Time ({x_var})')
    g._legend.set_title(leg_titles[params['hue']])
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle('Total thrombus platelet count',fontsize=14) 
    dfc.save_fig('Thrombus','tot_count')
    plt.show()

#------------------------------------------------------
def lineplot_count_reg(df,col='inside_injury'):# LINEPLOT WITH PLT COUNT OVER TIME & H
    params=params_choice('line')
    x_var=params['x']
    x_ls=[x_var]
    grouping_var_ls=[col]
    dfg=dfc.rolling_timecount(df,grouping_var_ls,x_ls)
    params.update({'data':dfg,'col':col,'col_order':cfg.var_order[col]})
    if col == 'injury_zone':
        params.update({'facet_kws':{'sharey': False}})
    g=sns.relplot(**params)
    g.set_ylabels('Platelet count')
    g.set_xlabels(f'Time ({x_var})')
    g._legend.set_title(leg_titles[params['hue']])
 #   plt.subplots_adjust(top=0.92)
    dfc.save_fig('Count',f'lineplot {col}')
    plt.show()

#-----------------------------------------------
def lineplot_newplts(df,col='inside_injury'):#LINEPLOT COUNT NEW, UNSTABLE & NET DIFF PLATELETS
    #BEHÖVER PUTSA PÅ GRAFEN OCH MINSKA STORLEKEN PÅ GRID 
    
    params=params_choice('line')
    df_new=df[(df.tracknr<3)&(df.frame>3)&(df.frame<190)]
    df_old=df[(df.tracknr>df.nrtracks-2)&(df.frame<190)]
    grouping_var=[col]
    x_var=params['x']
    x_ls=[x_var]
    dfg_new=dfc.rolling_timecount(df_new,grouping_var,x_ls)
    dfg_new['measure']='Number new'
    dfg_old=dfc.rolling_timecount(df_old,grouping_var,x_ls)
    dfg_old['measure']='Number unstable'
    dfg_growth=dfc.rolling_timecount(df,grouping_var,x_ls)
    dfg_growth['roll']=dfg_growth['diff']#dfg_new.roll-dfg_old.roll
    dfg_growth['measure']='Net Growth'
    dfg=pd.concat([dfg_new,dfg_old,dfg_growth],axis=0)
    params.update({'data':dfg,'row':'measure','col':col,'col_order':cfg.var_order[col],'aspect':1.25,'height':4})#'aspect'=1,'height=5',
    with sns.plotting_context("notebook"):
        g = sns.relplot(**params,facet_kws={'sharey': False})#
    g.set_ylabels('Count')
    g.set_xlabels(f'Time ({x_var})')
    g._legend.set_title(leg_titles[params['hue']])
    #g.fig.suptitle('Fraction newly recruited platelets',fontsize=25) 
    dfc.save_fig('lineplot','absolute turnover')
    plt.show()
#pms_line=dict(x='sec',hue='inh',ci=70, kind="line", hue_order=dfc.inh_order,
#                     height=4,aspect=1.25,legend=True)
#-----------------------------------------------
def lineplot_pltperc(df,col='inside_injury'): #LINEPLOT WITH FRACTION NEW, UNSTABLE & NET DIFF PLATELETS
    params=params_choice('line')
    df2=df[(df.tracknr<3)&(df.frame>3)&(df['sec']<580)]
    grouping_var=[col]
    x_var=params['x']
    x_ls=[x_var]
    dfg_perc_new=dfc.rolling_perc(df,df2,grouping_var,x_ls)
    dfg_perc_new['measure']='Fraction new'
    df2=df[(df.tracknr>(df.nrtracks-2))&(df['sec']<580)]
    dfg_perc_old=dfc.rolling_perc(df,df2,grouping_var,x_ls)
    dfg_perc_old['measure']='Fraction unstable'
    dfg_perc_growth=dfc.rolling_timecount(df,grouping_var,x_ls)
    dfg_perc_growth['roll']=(dfg_perc_growth['diff']/dfg_perc_growth['roll'])*100#dfg_new.roll-dfg_old.roll
    dfg_perc_growth['measure']='Fractional Growth'
    dfg=pd.concat([dfg_perc_new,dfg_perc_old,dfg_perc_growth],axis=0)
    params.update({'data':dfg,'row':'measure','col':col,'col_order':cfg.var_order[col]})   #'aspect'=1,'height=5',            
    with sns.plotting_context("notebook"):
        g = sns.relplot(**params,facet_kws={'sharey': False})#
    g.set_ylabels('Fraction')
    g._legend.set_title(leg_titles[params['hue']])
    #g.fig.suptitle('Fraction newly recruited platelets',fontsize=25) 
    dfc.save_fig('lineplot','relative turnover')
    plt.show()
    
    
#------------------------------------------------------
#LINEPLOTS MEANS OF VARIABLES
#------------------------------------------------------

# TIME ON X-AXIS 
#------------------------------------------------------
def lineplot_meantime(df,**xparams):# Generic function for lineplots showing means over time
    params={'hue':'inh','hue_order':dfc.inh_order,'col':'inside_injury','x':'sec','ci':95,
            'kind':'line','y':'stab','facet_kws':{'sharey': False}}#'legend':False
    params.update(xparams)
    return sns.relplot(data=df, aspect=1.25, height=6,**params)

def lineplot_stabmean(df,col_var):
    g=lineplot_meantime(df.dropna(subset=['stab']),y='stab',col=col_var, col_order=cfg.var_order[col_var])
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle('Stability',fontsize=25) 
    g.set_ylabels("Distance to closest \nneighbour in next frame ($\mu$m)")
    g.set_xlabels("Time (s)")
    dfc.save_fig('lineplot time','mean stab')
    plt.show()

def lineplot_contmean(df,col_var):
    g=lineplot_meantime(df.dropna(subset=['cont_s']),y='cont_s',col=col_var,facet_kws={'sharey': True}, col_order=cfg.var_order[col_var])
    #g = sns.relplot(x="time", y='roll', hue='inh',hue_order=dfc.inh_order,ci=70,col='position', data=df_grouped, kind="line",height=5,aspect=1,legend=True)
   # g.set(xlim=[0,600],xticks=[0,250,500],yticks=[0,200])
    #plt.legend(bbox_to_anchor=(0.4, 1), loc=2, borderaxespad=0., frameon=False, facecolor ='white', ncol=1,labels=dfc.inh_order)
    g.map(plt.axhline, y=0, ls="-", c=".5")
    g.set_ylabels("Contraction (nm/s)")
    g.set_xlabels("Time (s)")
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle('Mean platelet contraction',fontsize=25)
    dfc.save_fig('lineplot time','mean cont_s')
    plt.show()

def lineplot_conttotmean(df,col_var):
    g=lineplot_meantime(df.dropna(subset=['cont_tot']),y='cont_tot',col=col_var, col_order=cfg.var_order[col_var])
    #g = sns.relplot(x="time", y='roll', hue='inh',hue_order=dfc.inh_order,ci=70,col='position', data=df_grouped, kind="line",height=5,aspect=1,legend=True)
   # g.set(xlim=[0,600],xticks=[0,250,500],yticks=[0,200])
    #plt.legend(bbox_to_anchor=(0.4, 1), loc=2, borderaxespad=0., frameon=False, facecolor ='white', ncol=1,labels=dfc.inh_order)
    g.map(plt.axhline, y=0, ls="-", c=".5")
    g.set_ylabels("Total platelet contraction ($\mu$m)")
    g.set_xlabels("Time (s)")
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle('Mean total platelet contraction',fontsize=25)
    dfc.save_fig('lineplot time','mean cont_tot')
    plt.show()

def lineplot_nbad10mean(df,col_var):
    g=lineplot_meantime(df.dropna(subset=['nba_d_10']),col=col_var,y='nba_d_10', col_order=cfg.var_order[col_var])
    #plt.legend(bbox_to_anchor=(0.4, 1), loc=2, borderaxespad=0., frameon=False, facecolor ='white', ncol=1,labels=dfc.inh_order)
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle('Average distance to 10 closest platelets',fontsize=25) 
    g.set_xlabels("Time (s)")
    g.set_ylabels("Average distance ($\mu$m)")
    dfc.save_fig('lineplot','mean nba_d_10')
    plt.show()

def lineplot_c0mean(df,col_var):
    g=lineplot_meantime(df.dropna(subset=['c0_mean']),y='c0_mean',col=col_var,facet_kws={'sharey': True}, col_order=cfg.var_order[col_var])
    #plt.legend(bbox_to_anchor=(0.4, 1), loc=2, borderaxespad=0., frameon=False, facecolor ='white', ncol=1,labels=dfc.inh_order)
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle('Platelet calcium levels',fontsize=25) 
    g.set_ylabels("CAL520 fluorescence (AU)")
    g.set_xlabels("Time (s)")
    dfc.save_fig('lineplot','mean c0_mean')
    plt.show()

def lineplot_ymean(df,col_var):
    g=lineplot_meantime(df.dropna(subset=['ys']),y='ys',facet_kws={'sharey': True},col=None)
    #plt.legend(bbox_to_anchor=(0.4, 1), loc=2, borderaxespad=0., frameon=False, facecolor ='white', ncol=1,labels=dfc.inh_order)
    plt.subplots_adjust(top=0.85)
    #g.map(plt.axhline, y=0, ls="-", c=".5")
    g.fig.suptitle('Thrombus center of gravity, flow axis',fontsize=22) 
    g.set_ylabels("Mean position")
    g.set_xlabels("Time (s)")
    dfc.save_fig('lineplot','mean y')
    plt.show()

def lineplot_cacorrmean(df,col_var):
    #df['c0_corr']=calc_comp(df)
    g=lineplot_meantime(df.dropna(subset=['ca_corr']),col=col_var,y='ca_corr', col_order=cfg.var_order[col_var],facet_kws={'sharey': True})
    #plt.legend(bbox_to_anchor=(0.4, 1), loc=2, borderaxespad=0., frameon=False, facecolor ='white', ncol=1,labels=dfc.inh_order)
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle('Platelet corrected calcium levels',fontsize=25) 
    g.map(plt.axhline, y=0, ls="-", c=".5")
    g.set_ylabels("Corrected CAL520 fluorescence (AU)")
    g.set_xlabels("Time (s)")
    dfc.save_fig('lineplot','mean c0_corr')
    plt.show()

def lineplot_cacorrmean_height(df,col_var):
    #df['c0_corr']=calc_comp(df)
    g=lineplot_meantime(df.dropna(subset=['ca_corr']),y='ca_corr',col=col_var,#facet_kws={'sharey': True}
                        row='height',row_order=cfg.height_order, col_order=cfg.var_order[col_var])
    #plt.legend(bbox_to_anchor=(0.4, 1), loc=2, borderaxespad=0., frameon=False, facecolor ='white', ncol=1,labels=dfc.inh_order)
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle('Platelet corrected calcium levels',fontsize=25) 
    g.set_ylabels("Corrected CAL520 fluorescence (AU)")
    g.set_xlabels("Time (s)")
    g.set(ylim=[0,80])
    dfc.save_fig('lineplot','mean c0_corr')
    plt.show()

# OTHER VARIABLES ON X-AXIS
#------------------------------------------------------

# Generic function for lineplots showing means of two varibles
def lineplot_meanvar(df,x_var,y_var,**xparams):
    params={'hue':'inh','hue_order':dfc.inh_order,'col':'phase','x':x_var,'ci':95,
            'kind':'line','y':y_var}#'facet_kws':{'sharey': False},'legend':False
    params.update(xparams)
    return sns.relplot(data=df, aspect=1.25, height=6,**params)

# Plots with distance from center on x-axis
def lineplot_distcz_mov(dfg):
    dfg.loc[:,'dv_s']=dfg['dv']*1000/3.1
    y_var='dv_s'
    x_var='dist_cz'
    g=lineplot_meanvar(dfg,x_var,y_var,col=None)
    plt.subplots_adjust(top=0.80)
    g.fig.suptitle('Movement vs distance from center of injury',fontsize=22) 
    g.set_xlabels("Distance from center")
    g.set_ylabels("Movement (nm/sec)")
    #g.set_xticklabels(rotation=45)
    g.set(xlim=[0,125])
    dfc.save_fig('lineplot','dist_cz dv')
    plt.show()

def lineplot_distcz_nrtracks(dfg):
    dfg['nrtracks_s']=dfg.nrtracks*3.1
    y_var='nrtracks_s'
    x_var='dist_cz'
    g=lineplot_meanvar(dfg,x_var,y_var)
    g.map(plt.axvline, x=37.5, ls="--", c=".5")
    plt.subplots_adjust(top=0.80)
    g.fig.suptitle('Plt tracking time vs distance from center of injury',fontsize=22) 
    g.set_xlabels("Distance from center ($\mu$m)")
    g.set_ylabels("Average plt tracking time ('sec')")
    #g.set_xticklabels(rotation=45) 
    g.set(xlim=[0,125])
    dfc.save_fig('lineplot','dist_cz_nrtracks_s')
    plt.show()

def lineplot_distcz_stab(dfg):
    y_var='stab'
    x_var='dist_cz'
    g=lineplot_meanvar(dfg,x_var,y_var)
    plt.subplots_adjust(top=0.80)
    g.fig.suptitle('Stability vs distance from center of injury',fontsize=22) 
    g.set_xlabels("Distance from center ($\mu$m)")
    g.set_ylabels("Distance to closest \nneighbour in next frame ($\mu$m)")
    #g.set_xticklabels(rotation=45) 
    g.set(xlim=[0,125],ylim=[0,5])
    dfc.save_fig('lineplot','dist_cz stab')
    plt.show()

def lineplot_distcz_nba(dfg):
    y_var='nba_d_5'
    x_var='dist_cz'
    g=lineplot_meanvar(dfg,x_var,y_var)
    plt.subplots_adjust(top=0.80)
    g.fig.suptitle('Plt density vs distance from center of injury',fontsize=22) 
    g.set_xlabels("Distance from center")
    g.set_ylabels('Average distance to 5 closest platelets',fontsize=25)
    g.set(xlim=[0,125],ylim=[5,12])
    dfc.save_fig('lineplot','dist_cz nba_d_5')
    plt.show()

def lineplot_distcz_cacorr(dfg):
    #dfg['dv_s']=dfg.dv*1000/3.1
    y_var='ca_corr'
    x_var='dist_cz'
    g=lineplot_meanvar(dfg,x_var,y_var)

    plt.subplots_adjust(top=0.80)
    g.fig.suptitle('Calcium vs distance from center of injury',fontsize=22) 
    g.set_xlabels("Distance from center")
    g.set_ylabels('Intracellular calcium (% of max)',fontsize=25)
    g.set(xlim=[0,125],ylim=[0,80])
    dfc.save_fig('lineplot','dist_cz ca_corr')
    plt.show()

#Countplots
#------------------------------------------------------

def lineplot_count_isovol(df): #LINEPLOT WITH FRACTION NEW, UNSTABLE & NET DIFF PLATELETS
    df=df[(df.iso_vol<100)]
    params=dict(x='iso_vol',y='roll',hue='inh',hue_order=dfc.inh_order,ci=70, kind="line",  
                col='phase',col_order=cfg.phase_order,height=4,aspect=1.25,legend=True)
    x_var=params['x']
    grouping_var=['phase']
    dfg=dfc.rolling_timecount(df,grouping_var,[x_var])
    params.update({'data':dfg,'col':'phase','col_order':cfg.var_order['phase']})
    g=sns.relplot(**params)
    g.map(plt.axvline, x=37.5, ls="--", c=".5")
    g.set_ylabels('Platelet count')
    g.set_xlabels("Isovolumetric outer radius")
    #g.set(xlim=[0,100])#125
    g._legend.set_title(leg_titles[params['hue']])
 #   plt.subplots_adjust(top=0.92)
    dfc.save_fig(f'Count isovol',f'lineplot')
    plt.show()

def lineplot_countmov_isovol(df,grouping_var='movement'): #LINEPLOT WITH FRACTION NEW, UNSTABLE & NET DIFF PLATELETS
    df=df[(df.iso_vol<100)]
    params=dict(x='iso_vol',y='roll',hue=grouping_var,ci=70, kind="line", col='inh',col_order=dfc.inh_order, 
                row='phase',row_order=cfg.phase_order,height=4,aspect=1.25,legend=True)
    x_var=params['x']
    dfg=dfc.rolling_timecount(df,[grouping_var,'phase'],[x_var])
    params.update({'data':dfg})#,'col':'phase','col_order':cfg.var_order['phase']
    g=sns.relplot(**params)
    g.set_ylabels('Platelet count')
    g.set_xlabels("Distance from center")
    g._legend.set_title(leg_titles[grouping_var])
 #   plt.subplots_adjust(top=0.92)
    dfc.save_fig(f'Count {grouping_var}',f'lineplot')
    plt.show()
#------------------------------------------------------
# PLOTS WITH CATEGORICAL X-AXIS
#------------------------------------------------------


def catplot_meantime(df,**xparams):# General function for catplots showing means over time
    params={'hue':'inside_injury','hue_order':cfg.bol_order,'col':'inh','x':'minute','col_order':dfc.inh_order,'col_wrap':3,'ci':95,
            'kind':'point','y':'dvz_s'}
    params.update(xparams)
    return sns.catplot(data=df, height=4,**params)

#------------------------------------------------------
# MOVEMENT PER MIN
#------------------------------------------------------
def catplot_dvzmin(df,hue_var):
    df=dfc.scale_var(df,'dvz')
    g = catplot_meantime(df)#,hue=hue_var,hue_order=cfg.var_order[hue_var]
    g.set_ylabels("Movement (nm/sec)")
    g.map(plt.axhline, y=0, ls="-", c=".5")
    g.set_xlabels("Time (min)")
    #sns.plt.legend(loc='center left',bbox_to_anchor=(1,0.5))
    plt.subplots_adjust(top=0.80)
    g.fig.suptitle("Average movement in z axis",fontsize=30) 
    dfc.save_fig('catplot','dvz_min')
    plt.show()

def catplot_dvzminpos(df, hue_var):
    g = catplot_meantime(df,hue='position',hue_order=cfg.position_order)
    g.set_ylabels("Movement (nm/sec)")
    g.map(plt.axhline, y=0, ls="-", c=".5")
    g.set_xlabels("Time (min)")
    #sns.plt.legend(loc='center left',bbox_to_anchor=(1,0.5))
    plt.subplots_adjust(top=0.80)
    g.fig.suptitle("Average movement in z axis",fontsize=30) 
    dfc.save_fig('catplot','dvz_minpos')
    plt.show()

def catplot_dvyminpos(df, hue_var):
    df['dvy_s']=df.dvy*1000/3.1
    g = catplot_meantime(df,y='dvy_s')#,hue=hue_var,hue_order=cfg.var_order[hue_var]
    g.set_ylabels("Movement (nm/sec)")
    #g._legend.set_title("Platelet position relative laser injury")
    g.map(plt.axhline, y=0, ls="-", c=".5")
    #g.set(xticks=[0,3,6,9])
    g.set_xlabels("Time (min)")
    #sns.plt.legend(loc='center left',bbox_to_anchor=(1,0.5))
    plt.subplots_adjust(top=0.80)
    g.fig.suptitle("Average movement in y axis",fontsize=30) 
    dfc.save_fig('catplot','dvy_minpos')
    plt.show()

#------------------------------------------------------
# Catplots showing correlations between variables
#------------------------------------------------------

# Generic function for catplots showing correlations between two variables
def catplot_corrbin(df,bin_var,bin_order,y_var,**xparams):
    params={'hue':'inh','hue_order':dfc.inh_order,'x':bin_var,'order':bin_order,'ci':70,'kind':'point','y':y_var}
    params.update(xparams)
    g=sns.catplot(data=df, height=6,**params)
    g.set_xlabels("Fibrin fluorescence, percentile")
    g.set_xticklabels(rotation=45)
    return g

# Plots fraction unstable platelets as a function of fibrin fluorescence
def catplot_fibrin_percunstable(dfg,bin_var,bin_order):
    dfg1=dfg.groupby(['inh','inh_exp_id',bin_var]).agg({'pid':'count'})#'phase',
    dfg2=dfg[dfg.stab>3].groupby(['inh','inh_exp_id',bin_var]).agg({'pid':'count'})#'phase',
    dfg3=dfg2/dfg1*100
    dfg3=dfg3.reset_index()
    y_var='pid'
    g=catplot_corrbin(dfg3,bin_var,bin_order,y_var)
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle('Stability vs fibrin',fontsize=22) 
    g.set_ylabels("Fraction unstable platelets (%)") 
    dfc.save_fig('catplot','fibrin percent unstable')
    plt.show()

# Plots fraction unstable platelets as a function of fibrin fluorescence with phase as columns
def catplot_fibrin_percunstable_phases(dfg,bin_var,bin_order):
    dfg1=dfg.groupby(['inh','inh_exp_id','phase',bin_var]).agg({'pid':'count'})#'phase',
    dfg2=dfg[dfg.stab>3].groupby(['inh','inh_exp_id','phase',bin_var]).agg({'pid':'count'})#'phase',
    dfg3=dfg2/dfg1*100
    dfg3=dfg3.reset_index()
    y_var='pid'
    g=catplot_corrbin(dfg3,bin_var,bin_order,y_var,col='phase')
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle('Stability vs fibrin',fontsize=22) 
    g.set_ylabels("Fraction unstable platelets (%)")
    dfc.save_fig('catplot','fibrin percent unstable phases')
    plt.show()

# Plots movement in y axis as a function of fibrin fluorescence
def catplot_fibrin_dvy(dfg,bin_var,bin_order):
    dfg.loc[:,'dvy_s']=dfg['dvy']*1000/3.1
    y_var='dvy_s'
    g=catplot_corrbin(dfg,bin_var,bin_order,y_var)
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle('Sliding vs fibrin',fontsize=22) 
    g.set_ylabels("Sliding (nm/sec)")
    dfc.save_fig('catplot','fibrin dvz')
    plt.show()

def catplot_fibrin_mov(dfg,bin_var,bin_order):
    dfg['dv_s']=dfg['dv'].copy()*1000/3.1
    y_var='dv_s'
    g=catplot_corrbin(dfg,bin_var,bin_order,y_var,col='phase')
    plt.subplots_adjust(top=0.80)
    g.fig.suptitle('Movement vs fibrin',fontsize=22) 
    g.set_ylabels("Movement (nm/sec)")
    dfc.save_fig('catplot','fibrin dv')
    plt.show()

def catplot_fibrin_stab(dfg,bin_var,bin_order):
    dfg['stab_s']=dfg.stab#*1000#/3.1
    y_var='stab_s'
    g=catplot_corrbin(dfg,bin_var,bin_order,y_var,col='phase')
    plt.subplots_adjust(top=0.80)
    g.fig.suptitle('Stab vs fibrin',fontsize=22) 
    g.set_ylabels("Stab ($\mu$m)")
    dfc.save_fig('catplot','fibrin stab')
    plt.show()

def catplot_fibrin_cacorr(dfg,bin_var,bin_order):
    #dfg['stab_s']=dfg.stab#*1000#/3.1
    y_var='ca_corr'
    g=catplot_corrbin(dfg,bin_var,bin_order,y_var,col='phase')
    plt.subplots_adjust(top=0.80)
    g.fig.suptitle('Intracellular calcium vs Fibrin',fontsize=22) 
    #g.set_xlabels("Fibrin fluorescence")
    g.set_ylabels("Intracellular Calcium, % of max")
    #g.set_xticklabels(rotation=45)  
    dfc.save_fig('catplot','fibrin ca_corr')
    plt.show()

#---------------------------------------------------------------------------
# Heatmaps
#---------------------------------------------------------------------------
def def_global_col_heat(hue_var):
    global heat_col_var
    heat_col_var=hue_var

def do_heatmaps_count(df,**params):
    thr_regions=params['regions']#sorted(df[hue_var].unique().tolist(),reverse=True)
    for region in thr_regions:
        print(heat_col_var,region)
        dfg=df[df[heat_col_var]==region].copy()
        if params['groups_ls']:
            for groups in params['groups_ls']:
                params.update({'groups':groups,'region':region})
                heatmap_count(dfg,**params)
        else:
            params.update({'region':region})
            heatmap_count(dfg,**params)

def do_heatmaps_mean(df,**params):
    thr_regions=params['regions']#sorted(df[heat_col_var].unique().tolist(),reverse=True)
    #print('Mean variable value Heatmaps')
    for region in thr_regions:
        print(f'Measure:{heat_col_var},{region}')
        dfg=df[df[heat_col_var]==region].copy()
        if params['groups_ls']:
            for groups in params['groups_ls']:
                params.update({'groups':groups,'region':region})
                heatmap_mean(dfg,**params)
       
        else:
            params.update({'region':region})
            heatmap_mean(dfg,**params)
            
def do_heatmaps_perc(df,**params):
    thr_regions=params['regions']#sorted(df[hue_var].unique().tolist(),reverse=True)
    for region in thr_regions:
        print(f'Measure:{heat_col_var}, {region}')
        dfg=df[df[heat_col_var]==region].copy()
        if params['desc']=='new':
            dfg1=dfg[dfg.tracknr.isin([1,2])].copy()
        elif params['desc']=='unstable':
            dfg1=dfg[dfg.stab>3].copy()
        elif params['desc']=='contracting':
            dfg1=dfg[(dfg.movement=='contracting')].copy()
        elif params['desc']=='drifting':
            dfg1=dfg[(dfg.movement=='drifting')].copy()
        if params['groups_ls']:
            for groups in params['groups_ls']:
                params.update({'groups':groups,'region':region})      
                heatmap_perc(dfg,dfg1,**params)
        else:
            params.update({'region':region})
            heatmap_perc(dfg,dfg1,**params)

def heatmap_count(dfg,**params):
    axs=def_heatmap_axs(**params)#max_col,min_col,groups,orient
    if params['smooth']:
        for n,ax in enumerate(axs):
            dfg1=dfg.loc[(dfg.inh==params['groups'][n])]
            dfg1=df_count(dfg1,**params)
            #dfg1[dfg1.iloc[:,:]<count_threshold]=min_col
            plot_heatmap(dfg1,n,ax,**params)
    dfc.save_fig(heat_col_var+str(params['region']),'heatmap_count')
    plt.show()

def heatmap_mean(dfg,**params):#,center_color,max_col,min_col,groups,var,orient#groups=dfc.inh_order count_thr=0
    #inside=dfg.inside_injury.unique()
    axs=def_heatmap_axs(**params)
    count_thr=params['count_thr']
    for n,ax in enumerate(axs):
        dfgn=dfg[(dfg.inh==params['groups'][n])]       
        dfg_mean=df_mean(dfgn,**params)
        #dfg1=dfg1.where(count_mask, min_col)
        if count_thr>0:
            dfg_count=df_count(dfgn,**params)
            #count_mask=dfg_count.iloc[:,:]<count_threshold
            #dfg1=dfg1.groupby(['zled','tled','inh_exp_id']).mean()[var].reset_index()
            dfg_count=dfg_count.reindex_like(dfg_mean)
            if params['cmap']=='div':
                dfg_mean[dfg_count.iloc[:,:]<count_thr]=0
            else:
                dfg_mean[dfg_count.iloc[:,:]<count_thr]=params['vmin']
        plot_heatmap(dfg_mean,n,ax,**params)
    dfc.save_fig(params['var']+'_'+'heatmap_mean',heat_col_var+'_'+str(params['region']))
    plt.show()

def heatmap_perc(dfg,dfg_c,**params):#,center_color
    var=params['var']
    count_threshold=10
    #inside=dfg.inside_injury.unique()
    axs=def_heatmap_axs(**params)
    #dfg=led_bins(dfg)
    #dfg_c=led_bins(dfg_c)
    for n,ax in enumerate(axs):
        dfg1=dfg.loc[(dfg.inh==params['groups'][n])]
        dfg1_c=dfg_c.loc[(dfg_c.inh==params['groups'][n])]
        dfg1=dfg1.groupby(['zled','tled','inh_exp_id']).count()[var]#.reset_index()
        dfg1_c=dfg1_c.groupby(['zled','tled','inh_exp_id']).count()[var]#.reset_index()
        dfg1i=dfg1.reset_index()
        df_count=dfg1i.groupby(['zled','tled']).mean().reset_index()
        dfg1=dfg1_c/dfg1#dfg1_c[var]/dfg1[var]
        dfg1=dfg1.groupby(['zled','tled']).mean().reset_index()
        dfg1['count']=df_count[var]
        #dfg1['mask']=(dfg1['count']>count_threshold)|(dfg1['zled']>4)
        #dfg1.loc[((dfg1['count']<count_threshold)&(dfg1['zled']>4)),var]=np.nan
        #df_mask=dfg1.pivot(index='zled', columns='tled', values='mask')
        #mask=(df_mask['count']<count_threshold)&(df_mask['zled']>4)
        dfg1=dfg1.pivot(index='zled', columns='tled', values=var)
        dfg1=dfg1.fillna(params['vmin'])
        for arg in [params['smooth']]:
            dfg1=heatmap_filter(dfg1,arg)
            #print(dfg1)
        plot_heatmap(dfg1,n,ax,**params)
        #df_mask1=dfg1.copy()
        #df_mask1.loc[:,:]=min_col
        #print(df_mask)
        #print()
        #df_mask=(df_mask | dfg1>df_mask1)
        #df_mask=~df_mask
        #dfg1=dfg1[df_mask]#=min_col
        #dfg1=dfg1.fillna(min_col)
    dfc.save_fig(params['desc']+heat_col_var+str(params['region']),'heatmap_perc')
    plt.show()

def def_heatmap_axs(**params):#max_col,min_col,groups,orient
    #params_heatmap=dict(xticklabels=15, cmap="turbo",vmax=max_col,vmin=min_col)#,center=center_color)
    if params['orient']=='horizontal':
        sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 1})   
        n=len(params['groups'])
        if len(params['groups'])<4:
            fig, axs = plt.subplots(1, n, figsize=(n*6,2), constrained_layout=True)
        else:
            fig, axs = plt.subplots(2, m.ceil(n/2), figsize=(n*3,6), constrained_layout=True)
    if params['orient'] == 'vertical':
        sns.set_context("paper", font_scale=2, rc={"lines.linewidth": 2})   
        fig, axs = plt.subplots(n, 1, figsize=(5,n*2.5), constrained_layout=True)
    return (axs)



def df_count(dfg1,**params):
    var=params['var']
    dfg1=dfg1.groupby(['zled','tled','inh_exp_id']).count()[var].reset_index()
    dfg1=dfg1.groupby(['zled','tled']).mean().reset_index()
    dfg1=dfg1.pivot(index='zled', columns='tled', values=var)
    dfg1=dfg1.fillna(0)
    for arg in [params['smooth']]:#.values():#smooth:#
            dfg1=heatmap_filter(dfg1,arg)
    return dfg1

def df_mean(df,**params):
    dfg_mean=df.groupby(['zled','tled']).mean()[params['var']].reset_index()
    #df_count=dfgn.groupby(['zled','tled','inh_exp_id']).count()['pid'].reset_index()
    #dfg_mean['count']=df_count.groupby(['zled','tled']).mean().reset_index()['pid']
    #dfg1['mask']=(dfg1['count']>count_threshold)|(dfg1['zled']>c_thr)
    #dfg1=pd.concat([dfg_mean,df_count],axis=1)
    #dfg_mean.loc[((dfg_mean['count']<count_threshold)&(dfg_mean['zled']>4)),var]=np.nan
    dfg1=dfg_mean.pivot(index='zled', columns='tled', values=params['var'])
    dfg1=dfg1.fillna(params['vmin']) #Kan censurera helt om man tar bort denna
    #dfg1=dfg1.fillna(0)
    for arg in [params['smooth']]:
        dfg1=heatmap_filter(dfg1,arg)
    return dfg1

def heatmap_filter(dfg1,smooth):
    dfg_n=dfg1.to_numpy()
    if smooth == 'uniform':
        dfg_n=ndimage.uniform_filter(dfg_n, size=2)
    elif smooth == 'gauss':
        dfg_n=ndimage.gaussian_filter(dfg_n, sigma=2)
    elif smooth == 'gauss1D':
        dfg_n=ndimage.gaussian_filter1d(dfg_n, sigma=2)
    dfg1.loc[:,:]=dfg_n
    #dfg1=dfg1.fillna(0)
    return dfg1

def plot_heatmap(dfg1,n,ax,**params):
    sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 2})
    params['cmap']=cfg.cmaps[params['cmap']]
    if params['orient']=='horizontal':
        horizontal_heatmap(dfg1,n,ax,**params) 
                
    elif params['orient'] == 'vertical':
        vertical_heatmap(dfg1,n,ax,**params)

def vertical_heatmap(dfg1,n,ax,**params):
    heat_params = {key: value for key,value in params.items() 
                   if key in ['xticklabels','yticklabels','cmap','vmax','vmin']}
    if n == (len(params['groups'])-1):     
        sns.heatmap(dfg1, cbar=True,ax=ax,**params)#yticklabels=9, 
        ax.set_xlabel('Time (s)',fontsize=14)      
    else:
        sns.heatmap(dfg1, cbar=False,ax=ax,**params)#yticklabels=9,
    ax.invert_yaxis()
    ax.set_ylabel('Height ($\mu$m)',fontsize=14)
    ax.set_ylabel('')
    ax.set_xlabel('')
    #ax.set_title(groups[n])
    ax.annotate(params['groups'][n], xy=(3, 1), color='white',  xycoords='data',
    xytext=(0.8, 0.95), textcoords='axes fraction',
    horizontalalignment='right', verticalalignment='top',fontsize=20)


def horizontal_heatmap(dfg1,n,ax,**params):
    
    if n == (len(params['groups'])-1):
        heat_params = {key: value for key,value in params.items() 
                       if key in ['xticklabels','yticklabels','cmap','vmax','vmin']}
        sns.heatmap(dfg1, cbar=True,ax=ax,**heat_params)#yticklabels=8,
    else:  
        heat_params = {key: value for key,value in params.items() 
                       if key in ['xticklabels','cmap','vmax','vmin']}
        sns.heatmap(dfg1, yticklabels=False,cbar=False,ax=ax,**heat_params)      
    ax.invert_yaxis()
    ax.yaxis.tick_right() 
    ax.set_ylabel('')
    ax.set_xlabel('')
    #ax.set_title(groups[n])
    ax.annotate(params['groups'][n], xy=(3, 1), color='white',  xycoords='data',
    xytext=(0.8, 0.95), textcoords='axes fraction',
    horizontalalignment='right', verticalalignment='top',fontsize=20)

#---------------------------------------------------------------------------
# Mapping platelet positions
#---------------------------------------------------------------------------
def plt_map(df_obj,col_var,x_var,vmin,vmax): #Map of platelets at different time points coloured with name variable
    #plt.rcParams['image.cmap'] = 'viridis'#'coolwarm'#"turbo"    #plt.rcParams['image.cmap'] = 'jet_r'
    sns.set_style("white")
    #Set boundaries of plots #params={'col':'path','row':'c','hue':'c',}
    lims=['x_s', 'ys', 'zs']
    limsv=dict(x_s=(-100,100),ys=(-120,80),zs=(0,100))
    #limsv={}
    #for l in lims:
    #    limsv[l]=df_obj[l].min(), df_obj[l].max()   
    #Pick frames for visualization
    frames=[10,20,30,50,90,180]#pd.unique(df_obj.frame)[::20]+10
    ncols=3
    nrows=len(frames)
    #Set figure size  
    plt.figure(figsize=(ncols*4,nrows*4))
    #Choose plotting dimensions in graphs
    cols=[('x_s', 'ys','zs'), ('x_s', 'zs','ys'), ('ys', 'zs','x_s')]
    ### Set color variable name='cld'#name='stab'#name='c'#name='depth' #colorv=[1,2,4,8] #name='c2_max'
    #vmin=0 #vmax=30#vmax=10 #vmax=400
    for r, f in enumerate(frames):
        #sel_f=df_obj[df_obj.frame==f]
        sel_f=df_obj[df_obj.frame.isin(range(f-2,f+2))]
        for c, xy in enumerate(cols):
            sel_f.sort_values(by=xy[2])
            ax=plt.subplot2grid((nrows, ncols), (r, c))
            ax.scatter(sel_f[xy[0]], sel_f[xy[1]], alpha=0.7, c=sel_f[col_var],vmin=vmin,vmax=vmax,s=30, linewidth=0.1,cmap='coolwarm')#'bwr' 'coolwarm'TOG BORT , vmin=vmin, vmax=vmax,)
            ax.set_title('Time (sec): '+ str(np.round(sel_f[x_var].mean())),fontsize=12)#sel_f.time.mean()
            ax.set_ylim(limsv[xy[1]])
            ax.set_xlim(limsv[xy[0]])
            plt.xticks([])#-37.5,37.5
            plt.yticks([])#-37.5,37.5
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(labelsize=12)
            if xy[1]=='ys':
                circle=plt.Circle((0, 0), 37.5, alpha=0.4,fc='grey',edgecolor='black')#edgecolor='black',linewidth=7,fill='black'
                ax.add_patch(circle)
            else:
                ax.hlines(y=0, xmin=-38, xmax=38, linewidth=8, color='grey',alpha=0.8)
            sns.despine(top=True, right=True, left=True, bottom=True)
            #ax.ticklabel_format()
            #ax.set_axis_bgcolor('black')
    plt.tight_layout()
    dfc.save_fig(col_var,'plt_map')  

def plt_map2(df_obj,col_var,x_var,vmin,vmax): #Map of platelets at different time points coloured with name variable
    #plt.rcParams['image.cmap'] = 'viridis'#'coolwarm'#"turbo"    #plt.rcParams['image.cmap'] = 'jet_r'
    sns.set_style("white")
    #Set boundaries of plots #params={'col':'path','row':'c','hue':'c',}
    lims=['x_s', 'ys', 'zs']
    limsv=dict(x_s=(-100,100),ys=(-120,80),zs=(0,100))
    #limsv={}
    #for l in lims:
     #   limsv[l]=df_obj[l].min()-1, df_obj[l].max()+1   
    #Pick frames for visualization
    frames=[10,20,30,50,90,180]#pd.unique(df_obj.frame)[::20]+10
    ncols=len(lims)
    nrows=len(frames)
    size=ncols*nrows
    #Set figure size  
    #Choose plotting dimensions in graphs
    cols=[('x_s', 'ys','zs'), ('x_s', 'zs','ys'), ('ys', 'zs','x_s')]
    ### Set color variable name='cld'#name='stab'#name='c'#name='depth' #colorv=[1,2,4,8] #name='c2_max'
    #vmin=0 #vmax=30#vmax=10 #vmax=400
    for c, xy in enumerate(cols,0):
        fig=plt.figure(figsize=(6,nrows*6))#figsize=(4,30)
        #fig.set_title(inhibitor)
        gs=GridSpec(nrows,1)
        plot_nr=0
        for r, f in enumerate(frames):
            sel_f=df_obj[df_obj.frame.isin(range(f-2,f+2))]
            #sel_f=df_obj[df_obj.frame==f]
            sel_f.sort_values(by=xy[2])
            ax=fig.add_subplot(gs[plot_nr])
            ax.scatter(sel_f[xy[0]], sel_f[xy[1]], alpha=0.7, c=sel_f[col_var],vmin=vmin,vmax=vmax,s=60, linewidth=0.1,cmap='coolwarm' )#TOG BORT , vmin=vmin, vmax=vmax,)
            ax.set_title('Time (sec): '+ str(np.round(sel_f[x_var].mean())),fontsize=12)#sel_f.time.mean()
            ax.set_ylim(limsv[xy[1]])
            ax.set_xlim(limsv[xy[0]])
            plt.xticks([])#-37.5,37.5
            plt.yticks([])#-37.5,37.5
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(labelsize=12)
            if xy[1]=='ys':
                circle=plt.Circle((0, 0), 37.5, alpha=0.4,fc='grey',edgecolor='black')#edgecolor='black',linewidth=7,fill='black'
                ax.add_patch(circle)
            else:
                ax.hlines(y=0, xmin=-38, xmax=38, linewidth=8, color='grey',alpha=0.8)
            sns.despine(top=True, right=True, left=True, bottom=True)
            plot_nr+=1
                #ax.ticklabel_format()
                #ax.set_axis_bgcolor('black')
        gs.tight_layout(fig)
    #plt.tight_layout()
    dfc.save_fig(col_var,'plt_map')  
#---------------------------------------------------------------------------
# Mapping platelet trajectories
#---------------------------------------------------------------------------



#OBS! MÅSTE FIXA VARIABELN time SÅ ATT DEN FUNGERAR I FUNKTIONEN INNAN DENNA FUNKAR ATT KÖRA!!!"
def t_traj_mov(df,c_var='ca_corr',**xtra_params):#vmin,vmax
    df=df.sort_values(by=['pid'])
    sns.set_style("white")#sns.set_style("dark")# sns.set_style("white")
    plt.rcParams['image.cmap'] = 'coolwarm'
    plt.rcParams.update({'font.size': 22})
    params={'cols':3,'nrows':3,'hue':cfg.mov_class_order1,'vmin':0,'vmax':70,'time_bins':'phase','c_var':'tracknr'}
    params.update(xtra_params)
    for c, inhibitor in enumerate(pd.unique(df.inh),0):
        fig=plt.figure(figsize=(12,14))#figsize=(4,30)
        #fig.set_title(inhibitor)
        gs=GridSpec(params['nrows'],params['cols'])
        plot_nr=0
        for time in df['time_bin'].unique():
            for pop in params['hue']:
                plt_pop=df[(df.mov_phase==pop)&(df['time_bin']==time)&(df.inh==inhibitor)].copy()
                #pop_part=plt_pop.particle.unique()
                ax=fig.add_subplot(gs[plot_nr])
                
                plt_pop.sort_values(params['c_var'],ascending=False)
                plt.scatter(x=plt_pop.x_s, y=plt_pop.ys , c=plt_pop[c_var], s=10, alpha=0.7, cmap='viridis',vmax=params['vmax'], vmin=params['vmin'], linewidth=0)#'coolwarm'
                #plt.scatter(x=0, y=0, s=375,c='none', alpha=0.5, linewidth=40,edgecolor='black')#c='black',     
                circle=plt.Circle((0, 0), 37.5, alpha=0.4,fc='grey',edgecolor='black')#edgecolor='black',linewidth=7,fill='black'
                ax.add_patch(circle)
                plt.axis([-100, 100, -100, 100])
                plt.xlim(-100,100)  
                plt.ylim(-125,100)
                plt.xticks([])#-37.5,37.5
                plt.yticks([])#-37.5,37.5
                if pop==params['hue'][0]:
                    ax.set_ylabel(f'Time range: {np.round(time.left,-1)}-{np.round(time.right,-1)} s',fontsize=14)
                if plot_nr<3:
                    ax.set_title(pop)
                #if pop==plot[2]:
                #    plt.colorbar()
                plot_nr+=1
        sns.despine(top=True, right=True, left=True, bottom=True)
        fig.suptitle(inhibitor, fontsize=16)
        treatment=cfg.longtoshort_dic[inhibitor]
        dfc.save_fig(f'traj_map_{treatment}_','mov_class')
        plt.show()


#---------------------------------------------------------------------------
# Outlier detection plots
#---------------------------------------------------------------------------
def outliers_nrtracks(pc):
    test_var='nrtracks'
    pc_test=pc
    dfg=pc_test.groupby(['inh','path']).mean()[[test_var]].reset_index()
    outliers=[]
    for inh in dfg.inh.unique():
        dfg_inh=dfg[(dfg.inh==inh)].copy()
        outliers.append(dfg_inh[(np.abs(stats.zscore(dfg_inh[test_var])) > 1)]['path'])#['inh_id'])#Changed from 2!!!
    df_outliers=pd.concat(outliers,axis=0)
    dfg['outlier']=dfg.path.isin(df_outliers)
    dfg['value']=dfg[test_var]
    g=sns.catplot(data=dfg,y='value',x='inh',hue='outlier',height=5,aspect=3,kind='swarm',legend=False)
    g.set_xticklabels(rotation=45)
    dfc.save_fig('all variables','outliers nrtracks')        
    plt.show()
    
    return dfg[(dfg.outlier==True)],dfg[(dfg.outlier==False)]

def outliers_count(df):
    hue_order=['True','False','Both']
    df['tracked']=df.nrtracks>1
    dfg_count=df.groupby(['inh','exp_id','tracked']).count()[['pid']].reset_index()
    dfg_count1=df.groupby(['inh','exp_id']).count()[['pid']].reset_index().set_index(
        ['inh','pid']).sort_index().reset_index()
    dfg_count1['tracked']='Both'
    dfg=pd.concat([dfg_count,dfg_count1],axis=0)
    outliers=[]
    for inh in dfg.inh.unique():
        dfg_inh=dfg[(dfg.inh==inh)&(dfg.tracked=='True')].copy()
        outliers.append(dfg_inh[(np.abs(stats.zscore(dfg_inh.pid)) > 2)]['exp_id'])
    df_outliers=pd.concat(outliers,axis=0)
    dfg['outlier']=dfg.exp_id.isin(df_outliers)
    dfg['value']=dfg['pid']
    g=sns.catplot(data=dfg,y="value",x='tracked',col='inh',hue='outlier',col_wrap=3,height=4,kind='swarm')
    dfc.save_fig('all variables','outliers count')      
    plt.show()
    return dfg[(dfg.outlier==True)&(dfg.tracked=='True')]

def outliers_count1(df):
    dfg=df.groupby(['inh','path']).count()[['pid']].reset_index()#.set_index(#,'mouse','inj',
    outliers=[]
    for inh in dfg.inh.unique():
        dfg_inh=dfg[(dfg.inh==inh)].copy()
        outliers.append(dfg_inh[(np.abs(stats.zscore(dfg_inh.pid)) > 2)]['path'])
    df_outliers=pd.concat(outliers,axis=0)
    dfg['outlier']=dfg.path.isin(df_outliers)
    dfg['value']=dfg['pid']
    g=sns.catplot(data=dfg,y="value",x='inh',hue='outlier',height=5,aspect=3,kind='swarm')
    g.set_xticklabels(rotation=45)
    dfc.save_fig('all variables','outliers count') 
    plt.show()
    return dfg[(dfg.outlier==True)]

def outliers_fluo(pc):
    pc_test=pc[(pc.tracked==True)]#(pc.inside_injury==True)&(pc.height=='bottom')&
    dfg_fluo=pc_test.groupby(['inh','path']).mean()[['c0_mean','c0_max','c2_mean','c1_mean','c1_max']].reset_index()#.set_index(
    id_cols=['inh','path',]#'mouse','inj',
    melt_vars=['c0_mean','c1_mean','c2_mean',]#'c0_max','c1_max'
    #melt_vars=['c0_mean','c0_max','c2_mean','c1_mean','c1_max']
    dfg_fluo_long=dfg_fluo.melt(id_vars=id_cols,value_vars=melt_vars,var_name='Measure',value_name='Fluorescence')
    outliers=[]
    for inh in dfg_fluo_long.inh.unique():
            for measure in dfg_fluo_long.Measure.unique():
                dfg_=dfg_fluo_long[(dfg_fluo_long.inh==inh)&(dfg_fluo_long.Measure==measure)].copy()
                dfg_['outlier']=(np.abs(stats.zscore(dfg_.Fluorescence)) > 2)
                outliers.append(dfg_)
    try:
        df_outliers=pd.concat(outliers,axis=0)
    except ValueError: 
        print(pc_test)
    g=sns.catplot(data=df_outliers,y="Fluorescence",x='inh',hue='outlier',row='Measure',height=5,aspect=3,kind='swarm',sharey=False)
    g.set_xticklabels(rotation=45)
    plt.show()
    df_outliers['value']=df_outliers['Fluorescence']
    dfc.save_fig('all variables','outliers fluo')      
    return df_outliers[df_outliers.outlier==True]



#-----------------------------------------------------------------------------------------
# BAR GRAPHS WITH THROMBUS PLT COUNT AUC 
#------------------------------------------------------------------------

def boxplot_plt_count(df):

#params_bars={x:"inh",order:dfc.inh_order, y:"pid", ci:70, palette:"vlag",alpha:.8}
    params_dict=dict(x="inh",order=dfc.inh_order,y="pid", ci=70, alpha=.8)


    dfg=df.groupby(['inh','inh_exp_id']).count()[['pid']].reset_index()
    dfg1=df.groupby(['inh','inh_exp_id','inside_injury']).count()[['pid']].reset_index()
    dfg2=df.groupby(['inh','inh_exp_id','position']).count()[['pid']].reset_index()
    dfg3=df.groupby(['inh','inh_exp_id','mov_class']).count()[['pid']].reset_index()
    dfg4=df.groupby(['inh','inh_exp_id','movement']).count()[['pid']].reset_index()

    f = plt.figure(figsize=(10, 20))
    gs = f.add_gridspec(5, 1)

    ax = f.add_subplot(gs[0, 0])
    g = sns.barplot(data=dfg, **params_dict)
    ax.set_ylabel('Platelet count')

    ax = f.add_subplot(gs[1, 0])
    g = sns.barplot(data=dfg1, hue='inside_injury',hue_order=cfg.bol_order,**params_dict)
    ax.legend(title='Inside Injury', bbox_to_anchor=(1.3, 1))
    ax.set_ylabel('Platelet count')

    ax = f.add_subplot(gs[2, 0])
    g = sns.barplot(data=dfg2, hue='position',hue_order=cfg.position_order,**params_dict)
    ax.legend(bbox_to_anchor=(1.3, 1))
    ax.set_ylabel('Platelet count')

    ax = f.add_subplot(gs[3, 0])
    g = sns.barplot(data=dfg3, hue='mov_class',hue_order=cfg.mov_class_order1,**params_dict)
    ax.legend(bbox_to_anchor=(1.3, 1))
    ax.set_ylabel('Platelet count')

    ax = f.add_subplot(gs[4, 0])
    g = sns.barplot(data=dfg4, hue='movement',hue_order=cfg.movement_order1,**params_dict)
    ax.legend(bbox_to_anchor=(1.3, 1))
    ax.set_ylabel('Platelet count')
    
    
    dfc.save_fig('count','boxplot')

#-----------------------------------------------------------------------------------------
# BAR GRAPHS WITH DISTRIBUTION OF PLATELETS IN DIFFERENT SUBPOPULATIONS
#-----------------------------------------------------------------------------------------

def boxplot_plt_fraction(df):
    
    params_dict=dict(x="inh",order=dfc.inh_order,y="pid", ci=70, alpha=.8)
    
    dfg1=df.groupby(['inh','inh_exp_id','inside_injury']).agg({'pid':'count'})
    dfg1 = dfg1.groupby(level=[0,1]).apply(lambda x:100 * x / float(x.sum())).reset_index()
    dfg2=df.groupby(['inh','inh_exp_id','position']).agg({'pid':'count'})
    dfg2 = dfg2.groupby(level=[0,1]).apply(lambda x:100 * x / float(x.sum())).reset_index()
    dfg3=df.groupby(['inh','inh_exp_id','mov_class']).agg({'pid':'count'})
    dfg3 = dfg3.groupby(level=[0,1]).apply(lambda x:100 * x / float(x.sum())).reset_index()
    dfg4=df.groupby(['inh','inh_exp_id','movement']).agg({'pid':'count'})
    dfg4 = dfg4.groupby(level=[0,1]).apply(lambda x:100 * x / float(x.sum())).reset_index()

    f = plt.figure(figsize=(12, 15))
    gs = f.add_gridspec(4, 1)

    ax = f.add_subplot(gs[0, 0])
    g = sns.barplot(data=dfg1, hue='inside_injury',hue_order=cfg.bol_order,**params_dict)
    ax.legend(title='Inside Injury', bbox_to_anchor=(1.2, 1))
    ax.set_ylabel('Fraction (%)')

    ax = f.add_subplot(gs[1, 0])
    g = sns.barplot(data=dfg2, hue='position',hue_order=cfg.position_order,**params_dict)
    ax.legend(bbox_to_anchor=(1.2, 1))
    ax.set_ylabel('Fraction (%)')

    ax = f.add_subplot(gs[2, 0])
    g = sns.barplot(data=dfg3, hue='mov_class',hue_order=cfg.mov_class_order1,**params_dict)
    ax.legend(bbox_to_anchor=(1.05, 1))
    ax.set_ylabel('Fraction (%)')

    ax = f.add_subplot(gs[3, 0])
    g = sns.barplot(data=dfg4, hue='movement',hue_order=cfg.movement_order1,**params_dict)
    ax.legend(bbox_to_anchor=(1.05, 1))
    ax.set_ylabel('Fraction (%)')
    
    dfc.save_fig('fraction','boxplot')