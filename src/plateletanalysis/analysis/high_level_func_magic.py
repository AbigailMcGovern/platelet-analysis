#-------------------------------------------------------------------------------------------------------------
# Master functions that automatically load dataframes and execute plots
#-------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np

from .. import config as cfg
from . import data_func_magic as dfc
from . import plot_func as plot
#import menu_func as mfc
#import calc_func as ca

#-------------------------------------------------------------------------------------------------------------
def masterfunc(df,thr_reg_var,time_var,analysis_):
    cfg.settings()
    #['timecounts','timemeans','varcorr','stats','outliers','heatmaps','custom']
    choice_made=False
    if 'timecounts' in analysis_:
        run_timecountplots(df,thr_reg_var)
        choice_made=True
    if 'timemeans' in analysis_:
        run_timemeanplots(df,thr_reg_var)
        choice_made=True
    if 'varcorr' in analysis_:
        run_catplotmeans(df,thr_reg_var)
        choice_made=True
    if 'stats' in analysis_:
        do_stats(df,thr_reg_var,time_var)
        choice_made=True
    if 'outliers' in analysis_:
        run_outliers(df)
        choice_made=True
    if 'heatmaps' in analysis_:
        run_heatmaps(df,'inside_injury')#You can change the region variable here!!!!
        choice_made=True
    if 'traj' in analysis_:
        run_traj(df)#You can change the region variable here!!!!
        choice_made=True
    if choice_made:
        print('Finished analysis')
    else:
        print('No analysis chosen.')
    

#-------------------------------------------------------------------------------------------------------------
# Master function that automatically runs a set of timecount lineplots for treatments defined in inh_order
#-------------------------------------------------------------------------------------------------------------

def run_timecountplots(df,xtra_var):
    print(f'{73 * "-"}\nStarting analysis of platelet counts over time\n{73 * "-"}')
    plot.lineplot_count_indexp(df,xtra_var)
    plot.lineplot_count_all(df)
    plot.lineplot_count_reg(df,col=xtra_var)
    if xtra_var != 'inside_injury':
        plot.lineplot_count_reg(df,col='inside_injury')
    if xtra_var != 'position':
        plot.lineplot_count_reg(df,col='position')
    plot.lineplot_newplts(df,col=xtra_var)  
    plot.lineplot_pltperc(df,col=xtra_var)

#-------------------------------------------------------------------------------------------------------------
# Master function that automatically runs a set of time-mean lineplots for treatments defined in inh_order
#-------------------------------------------------------------------------------------------------------------

def run_timemeanplots(df,col_var):
    print(f'{73 * "-"}\nStarting analysis of variable means over time\n{73 * "-"}')
    columns_=df.columns.tolist()
    
    if 'stab' in columns_:
        plot.lineplot_stabmean(df,col_var)
    if 'cont_s' in columns_:
        plot.lineplot_contmean(df,col_var)
    if 'cont_tot' in columns_:
        plot.lineplot_conttotmean(df,col_var)
    if 'nba_d_10' in columns_:
        plot.lineplot_nbad10mean(df,col_var)
    if 'c0_mean' in columns_:
        plot.lineplot_c0mean(df,col_var)
    if 'ca_corr' in columns_:
        plot.lineplot_cacorrmean(df,col_var)
    if 'height' in columns_ and 'ca_corr' in columns_:
        plot.lineplot_cacorrmean_height(df,col_var)
    if 'y_s' in columns_:
        plot.lineplot_ymean(df,col_var)

#-------------------------------------------------------------------------------------------------------------
# Master function that automatically runs a set of time-mean catplots for treatments defined in inh_order
#-------------------------------------------------------------------------------------------------------------

def run_catplotmeans(df,hue_var):
    
    columns_=df.columns.tolist()
    
    
    # Movements over time
    print(f'{73 * "-"}\nStarting analysis of platelet movements over time\n{73 * "-"}')
    plot.catplot_dvzmin(df, hue_var)
    plot.catplot_dvzminpos(df, hue_var)
    plot.catplot_dvyminpos(df, hue_var)

    #Fibrin vs other variables
    if 'c1_mean' in columns_:
        catplots_fibrin(df)
    
    #Dist_cz vs other variables
    if 'dist_cz' in columns_:

        lineplots_distc(df)
        lineplots_isovol(df)
        

# Calculates fibrin quantile bins and executes plots comparing calcium fluorescence with other measures
#-------------------------------------------------------------------------------------------------------------
def catplots_fibrin(df):
    #Binning
    dfg=df[(~df.inh.isin(['Bivalirudin','PAR4-/- + biva']))&(df.height=='bottom')].copy()
    binned_var='c1_mean'
    bins=10
    #dfg,bin_var,bin_order=qbinning_labels(dfg,binned_var,bins)
    dfg,bin_var,bin_order=dfc.qbinning_quant(dfg,binned_var,bins)
    dfg=dfc.phase_var(dfg)
    
    #Fibrin plots
    print(f'{73 * "-"}\nStarting analysis of fibrin fluorescence in core \n{73 * "-"}')
    plot.catplot_fibrin_percunstable(dfg,bin_var,bin_order)
    plot.catplot_fibrin_dvy(dfg,bin_var,bin_order)
    plot.catplot_fibrin_percunstable_phases(dfg,bin_var,bin_order)
    plot.catplot_fibrin_mov(dfg,bin_var,bin_order)
    plot.catplot_fibrin_stab(dfg,bin_var,bin_order)
    plot.catplot_fibrin_cacorr(dfg,bin_var,bin_order)

# Lineplots with distance from center on x-axis
#-------------------------------------------------------------------------------------------------------------
def lineplots_distc(dfg):
    #Rounds distance to closest integer
    print(f'{73 * "-"}\nStarting analysis of distance from center \n{73 * "-"}')
    dfg['dist_cz']=np.round(dfg.loc[:,'dist_cz'].copy(),decimals=0)
    dfg=dfg[dfg.dist_cz<100].copy()
    #dfg = dfg.astype({'dist_cz': int})
    
    #binned_var='dist_cz'
    #bins=20
    #dfg,bin_var,bin_order=qbinning_labels(dfg,binned_var,bins)
    #dfg,bin_var,bin_order=qbinning_quant(dfg,binned_var,bins)
    
    dfg=dfc.phase_var(dfg)
    
    #dist_c plots
    plot.lineplot_distcz_mov(dfg)
    plot.lineplot_distcz_stab(dfg)
    plot.lineplot_distcz_nrtracks(dfg)
    plot.lineplot_distcz_nba(dfg)
    plot.lineplot_distcz_cacorr(dfg)

# Lineplots with isovolumetric bins of distc_z on x-axis
#-------------------------------------------------------------------------------------------------------------
def lineplots_isovol(dfg):
    #Rounds distance to closest integer
    print(f'{73 * "-"}\nStarting analysis of isovolumetric layers \n{73 * "-"}')
    dfg=dfg[dfg['dist_cz']<125].copy()
    dfg=dfc.isovol_bin_var(dfg)
    dfg=dfc.phase_var(dfg)
    plot.lineplot_count_isovol(dfg)
    plot.lineplot_countmov_isovol(dfg)
#-------------------------------------------------------------------------------------------------------------
# Master function that automatically runs a set of heatmaps for treatments defined in inh_order
#-------------------------------------------------------------------------------------------------------------

def run_heatmaps(df,hue_var):
    
    
    plot.def_global_col_heat(hue_var)
    params_heatmap=plot.params_choice(choice='heat')
    ls_hue=[hue_var]

    df=dfc.led_bins_var(df)
    df['down']=-df['dvz_s']
    regions=sorted(df[hue_var].unique().tolist(),reverse=True)

    if len(dfc.inh_order)>3:
        group1,group2=dfc.chunks(dfc.inh_order,2)
        groups_ls=[group1,group2]
        params_heatmap.update({'group1':group1,'group2':group2,'groups':group1,'groups_ls':groups_ls})
    else:
        group1=dfc.inh_order
        group2=False
        groups_ls=False
        groups=dfc.inh_order
        params_heatmap.update({'group1':group1,'groups':group1,'groups_ls':False})

# Count heatmaps
#-------------------------------------------------------------------------------------------------------------
    print(f'{73 * "-"}\nStarting analysis of Platelet count heatmaps \n{73 * "-"}')
    params_heatmap.update({'vmax':100,'vmin':0,'groups':group1,'var':'pid','regions':regions})
    plot.do_heatmaps_count(df,**params_heatmap)#hue_var,


# Mean heatmaps
#-------------------------------------------------------------------------------------------------------------    
    print(f'{73 * "-"}\nStarting analysis of mean variable heatmaps \n{73 * "-"}')

    heat_mean_dic={
        0:{'var':'ca_corr','v':{'vmax':80,'vmin':10}},
        1:{'var':'dv','v':{'vmax':2,'vmin':0.3}},
        2:{'var':'nba_d_5','v':{'vmax':10,'vmin':6}},
        3:{'var':'nba_d_10','v':{'vmax':15,'vmin':8}},
        4:{'var':'nrtracks','v':{'vmax':170,'vmin':30}},
        5:{'var':'cont_s','v':{'vmax':200,'vmin':0}},
        6:{'var':'cont_s','v':{'vmax':200,'vmin':-100}},
        7:{'var':'down','v':{'vmax':150,'vmin':-50}},
         }
    params_heatmap.update({'count_thr':0})
    for n in heat_mean_dic.keys():
        heat_dic=heat_mean_dic[n]
        print('Mean values for:',heat_dic['var'])
        params_heatmap.update({'var':heat_dic['var'],'vmax':heat_dic['v']['vmax'],'vmin':heat_dic['v']['vmin']})
        plot.do_heatmaps_mean(df,**params_heatmap)#hue_var,
    

# Perc heatmaps
#-------------------------------------------------------------------------------------------------------------    
    print(f'{73 * "-"}\nStarting analysis of Percentile Count heatmaps \n{73 * "-"}')
    heat_per_dic={
        0:{'desc':'new','v':{'vmax':0.7,'vmin':0}},
        1:{'desc':'unstable','v':{'vmax':0.7,'vmin':0}},
        2:{'desc':'contracting','v':{'vmax':0.6,'vmin':0}},
        3:{'desc':'drifting','v':{'vmax':0.7,'vmin':0}},
              }
    
    var='pid'
    params_heatmap.update({'var':var,'count_thr':10})
    for n in heat_per_dic.keys():
        heat_dic=heat_per_dic[n]
        print('Fraction:',heat_dic['desc'])
        params_heatmap.update({'vmax':heat_dic['v']['vmax'],'vmin':heat_dic['v']['vmin'],'desc':heat_dic['desc']})
        plot.do_heatmaps_perc(df,**params_heatmap)#hue_var,



def run_heatmaps_count(df,hue_var):
    hues_ls=df[hue_var].unique().tolist()
    params_heatmap=plot.create_params()['heatmap']
    params_heatmap.update({})

#-------------------------------------------------------------------------------------------------------------    
# Trajectories
#-------------------------------------------------------------------------------------------------------------    

def run_traj(df):
    df=df
    


# ---------------------------------------------------------------------------
# Quality Control & Outlier detection
# ---------------------------------------------------------------------------
def run_outliers(df):
    inh_order=dfc.expseries_menu()
    df_count=plot.outliers_count1(df)
    df_count.insert(0,'measure','count')
    df_fluo=plot.outliers_fluo(df)
    df_fluo.insert(0,'measure','fluo')
    df_nrtracks,df_inliers=plot.outliers_nrtracks(df)
    df_nrtracks.insert(0,'measure','nrtracks')
    df_inliers.insert(0,'measure','no outlier')
    df_outliers=pd.concat([df_count,df_fluo,df_nrtracks],axis=0)
    df_outliers=df_outliers.reset_index()
    #outliers_ls=df_outliers_menu(df_outliers)
    df_outliers=dfc.df_outliers_menu(df_outliers,df_inliers)
    
        
    #return df_outliers
  
#---------------------------------------------------------------------------
# Statistics 
#---------------------------------------------------------------------------
def do_stats(df,thr_reg_var,time_var):
    
    #measure=int(input("Do you want to calculate statistics on:\n\n(1) Plts Counts\n(2) Means of variables\n"))
    
    #if measure == 1:
        #df,xtra_vars=dfc.build_df_statcounts(inh_order)
    xtra_vars=dfc.xtravars_stat(thr_reg_var,time_var)
    df_desc,df_tests=dfc.stats_counts(df,xtra_vars)
    test_var='plt_count'
    
    #print(df_desc,df_tests)
    dfc.save_table(df_desc,'Descriptive stats'+'_'+test_var)
    dfc.save_table(df_tests,'Statistical tests'+'_'+test_var)
    return df_desc,df_tests



