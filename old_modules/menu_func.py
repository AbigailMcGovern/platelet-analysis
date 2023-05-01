import pandas as pd
from pathlib import Path
from magicgui import magicgui
#from numpy import False_
from .. import config as cfg
from . import data_func_magic as dfc
from . import high_level_func_magic as hlc

#---------------------------------------------------------------------------
# Some global variables for use in magicgui
#---------------------------------------------------------------------------


controls_=['_ctrl_', '_saline_','_veh-mips_', '_salgav-veh_','_salgav_','_veh-sq_','_c2actrl_','_df_demo_']
Controls_=[cfg.shorttolong_dic[ctrl] for ctrl in controls_]
Controls_=Controls_+['None']
treatments_=['_biva_','_cang_','_mips_','_asa_','_asa-veh_','_sq_','_cmfda_','_par4--_','_par4+-_','_par4-+_','_par4--biva_','_c2akd_']
Treatments_=[cfg.shorttolong_dic[treat] for treat in treatments_]
Treatments_=Treatments_+['None']
Analysis_=[ 
    ('Plt counts over time', 'timecounts'), 
    ('Means of variables over time','timemeans'), 
    ('Plots showing dependencies between two variables','varcorr'),
    ('Statistical Calculations','stats'),
    ('Quality control & Outlier Detection','outliers'),
    ('Heatmaps', 'heatmaps'), 
    ('Trajectories & maps','traj'),
    ('Customized functions','custom' ),
    ]

#---------------------------------------------------------------------------
# Menu function created with MagicGui
#---------------------------------------------------------------------------

@magicgui(
    Controls = {'choices':Controls_,'allow_multiple':True},
    Treatments = {'choices':Treatments_,'allow_multiple':True},
    max_thr={"widget_type": "Slider", "min": 1, "label":"Upper tracking threshold"},
    min_thr={"widget_type": "Slider", "max": 200, "label":"Lower tracking threshold"},

    Save_figs={
        "widget_type": "RadioButtons",
        "orientation": "horizontal",
        "choices": [("Yes", True), ("No", False)],
        "label" : "Save Figures & Tables?"
    },
    File_formats={ 
        "widget_type": "RadioButtons",
        "orientation": "horizontal",
        #"choices": [(".png", 1), (".svg", 2),("both", 3)],
        "choices": [(".png"), (".svg"),("both")],
    },
    include_names={
        "widget_type": "RadioButtons",
        "orientation": "horizontal",
        "choices": [("Yes", True), ("No",False)],
        "label":"Include treatment names in filenames?",
    },
    del_outliers={
        "widget_type": "RadioButtons",
        "orientation": "horizontal",
        "choices": [("Yes", True), ("No", False)],
        "label":"Remove Outliers?",
    },
    file_folder={
        "name":"Folder_path",
        "label": "Choose a Folder for files:",
        "mode":'d'
        },  
    
    Analysis = {'choices':Analysis_,'allow_multiple':True},

    thr_reg_var = {
        "widget_type": "RadioButtons",
        "orientation": "horizontal",
        'choices': cfg.thr_reg_vars,
        "label":"Select thrombus region \nvariable to use in analysis:",
    },
    time_var = {
        "widget_type": "RadioButtons",
        "orientation": "horizontal",
        'choices': cfg.time_vars,
        "label":"Select xtra time \nvariable to use in analysis:",
    },
    call_button="Run analysis",
    layout="vertical",
    persist=True,
)
def input( Controls = ['Saline'], 
                Treatments = ['Cangrelor'], 
                max_thr = 200, min_thr = 1, 
                Save_figs = False, 
                File_formats = ".png", 
                include_names = True, 
                del_outliers = True, 
                file_folder = Path.home(), 
                Analysis = ['timecounts'], 
                thr_reg_var='injury_zone',
                time_var='phase'
                ):
    global results_folder, inh_order
    inh_order=[]
    results_folder=file_folder
    if Controls:
        if 'None' not in Controls:
            inh_order = Controls#.append(ctrl for ctrl in Controls)
        #if 'Demo Injuries' in Controls:
    if Treatments:
        if 'None' not in Treatments:
            inh_order += Treatments
    print(inh_order)
    #inh_order.append(treat for treat in Treatments)
    #vars=[(keys,values) for keys,values in locals()]
    #print(locals().keys(),locals().values())
    #return locals().items()
    #if 'custom' in Analysis:
    #    return df

@input.called.connect
def start_processing():
    global save_figs, results_folder,plot_formats,save_inh_names
    save_figs = input.Save_figs.value
    #results_folder = input.file_folder.value
    plot_formats = input.File_formats.value
    save_inh_names = input.include_names.value
    
    input.close()
    print(f'{73 * "-"}\nRun started, loading dataframe\n{73 * "-"}')
    print('Analysis value',input.Analysis.value)
    df_var_list=make_varlist(input.Analysis.value,input.thr_reg_var.value)
    df=dfc.build_df_lists(df_var_list,inh_order)
    df_cols=df.columns.tolist()
    if 'nrtracks' in df_cols:    
        df=df.loc[(df['nrtracks']>input.min_thr.value)&(df['nrtracks']<input.max_thr.value),:]
    if input.del_outliers.value:
        outliers=pd.read_csv('df_outliers.csv')
        df=df[~df.path.isin(outliers.path)]
    df=df.reset_index(drop=True)
    
    if input.thr_reg_var.value not in df_cols:
            df=dfc.add_xtravar(df,input.thr_reg_var.value)
    if input.time_var.value not in df_cols:
            df=dfc.add_xtravar(df,input.time_var.value)
    print(f'{73 * "-"}\nDataframe loaded, starting data analysis\n{73 * "-"}')
    hlc.masterfunc(df,input.thr_reg_var.value,input.time_var.value,input.Analysis.value)
    #input_var_dic=locals()
    print(f'{73 * "-"}\nAnalysis Completed\n{73 * "-"}')
    

    
    #func_b.input.value = value
    return df

def make_varlist( runs = ['timecounts'],thr_reg_var='inside_injury'): #Testa att göra om var_ls_ till tuple
    #print(runs)
    df_var_list=[['path', 'inh','particle']]
    print(runs)
    timecount_vars_=['frame','time','nrtracks','tracknr','inside_injury','position','dist_cz']#,
    timemean_vars_=['ys','position','inside_injury', 'height','frame','time','stab', 'dist_cz',
    'dvz','cont_s', 'cont_tot','ca_corr','c0_mean', 'c1_mean','nba_d_10','nrtracks','tracknr']
    varcorr_vars_ = ['position','inside_injury','height','dist_cz','frame','time','minute', 
    'dvy', 'dv','stab','mov_class', 'movement','dvz','cont_s','cont_tot', 'c0_mean', 'c1_mean', 
    'ca_corr','nba_d_5','nba_d_10','nba_d_15','nrtracks','tracknr']
    stat_vars_ = ['frame','time','nrtracks']
    outlier_vars_ = ['c0_mean', 'c0_max','c1_mean', 'c1_max','c2_mean', 'c2_max','nrtracks','tracknr','tracked']
    heatmap_vars_= ['zs','dist_c','depth','frame','time', 'dvy', 'dv','stab','mov_class', 'movement','dvz_s', 
    'cont_s','cont_tot','ca_corr', 'c0_mean','c1_mean','nba_d_5','nba_d_10','nrtracks','tracknr','exp_id','inside_injury']
    traj_vars_=['frame','time','x_s','ys','zs', 'dvx','dvy','dvz', 'ca_corr','c1_mean','cont','cont_tot','mov_class','movement','stab','tracknr','depth',]
    custom_vars_=['all_vars']

    run_names_=['timecounts','timemeans','varcorr','stats','outliers','heatmaps','traj','custom']
    var_ls_=[timecount_vars_,timemean_vars_,varcorr_vars_,stat_vars_,outlier_vars_,heatmap_vars_,traj_vars_,custom_vars_,]
    run_var_dic=dict(zip(run_names_, var_ls_))
    
    df_var_list += [value for key,value in run_var_dic.items() if key in runs]


    col_list = [item for sublist in df_var_list for item in sublist]
    
    if thr_reg_var in cfg.old_xtra_vars_:
        col_list.append(thr_reg_var)
    elif thr_reg_var in cfg.new_xtra_vars_:
        if thr_reg_var=='quadrant':
            new_vars=['x_s','ys']
            col_list+=new_vars
        elif thr_reg_var=='quadrant1':
            new_vars=['x_s','ys','zs']
            col_list+=new_vars
        elif thr_reg_var=='injury_zone':
            new_vars=['dist_cz']
            col_list+=new_vars


    #list(dict.fromkeys(items))
    col_list=list(dict.fromkeys(col_list))
    return col_list


    
