import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns
import warnings
import os

DIR_PATH = os.path.join(os.path.dirname(__file__), 'data')

#---------------------------------------------------------------------------
#df_path = Dataframe specifyting paths for files stored on computer
#----------------------------------------------------------------------
df_paths  = os.path.join(DIR_PATH, 'file_paths.csv')
df_paths = pd.read_csv(df_paths)
#df_paths=pd.read_csv('file_paths.csv')
#df_paths=pd.read_csv('file_paths - laptop.csv')
#---------------------------------------------------------------------------
#Lists and dictionaries of short and long treatment names
#Short names are used in file names and long names are used in graphs etc
#----------------------------------------------------------------------

cmaps={'div':'spectral','seq':'plasma'}#'viridis'
saline_ =['_saline_','_biva_','_cang_']
PAR4_ =['_ctrl_','_par4--_','_par4+-_','_par4-+_','_par4--biva_','_biva_']
MIPS_ =['_mips_','_veh-mips_','_asa-veh_','_salgav-veh_']
ASA_ =['_salgav_','_asa_']
SQ_ =['_veh-sq_','_sq_']
CMFDA_ =['_ctrl_','_cmfda_']
All_=['_biva_','_cang_','_ctrl_',
                '_mips_','_saline_','_sq_','_veh-mips_',
                '_veh-sq_','_par4+-_','_par4-+_','_par4--biva_',
                 '_par4--_','_asa-veh_','_asa_', '_salgav-veh_',
                 '_salgav_'
               ]
all_demo_= ['_saline_', '_biva_', '_cang_', '_veh-sq_', '_sq_', '_asa-veh_', '_asa_', '_veh-mips_', '_mips_', '_ctrl_', '_par4--_', '_par4--biva_', '_salgav-veh_', '_salgav_'] 
saline_demo_=['_saline_','_biva_','_cang_']
sq_demo_=['_veh-sq_', '_sq_']
asa_demo_=['_asa-veh_', '_asa_']
par4_demo_=['_ctrl_', '_par4--_', '_par4--biva_']
mips_demo_=['_mips_','_veh-mips_']
salgav_demo_=['_salgav-veh_', '_salgav_']
demo_ls_ls_=[saline_demo_,sq_demo_,asa_demo_,par4_demo_,mips_demo_,salgav_demo_]

simple_=['_par4--_','_cang_']


shortnames_ =['_biva_','_cang_','_cmfda_','_ctrl_',
                '_mips_','_saline_','_sq_','_veh-mips_',
                '_veh-sq_','_par4+-_','_par4-+_','_par4--biva_',
                 '_par4--_','_asa-veh_','_asa_', '_salgav-veh_',
                 '_salgav_','_c2actrl_','_c2akd_','_df_demo_',
               ]
longnames_ =['Bivalirudin','Cangrelor','CMFDA','Control',
                 'MIPS','Saline','SQ','Vehicle MIPS',
                 'Vehicle SQ','PAR4+/-','PAR4-/+','PAR4-/- + biva',
                 'PAR4-/-','ASA + Vehicle','ASA','Salgav + Vehicle',
                 'Salgav','C2alpha+','C2alpha-','Demo Injuries'
                ]

expseries_listnames=['Saline cohort','Thromin-PAR4 cohort','MIPS cohort','ASA cohort','SQ cohort','CMFDA cohort',
                     'All Treatments','Simple']
treatments_=[saline_,PAR4_,MIPS_,ASA_,SQ_,CMFDA_,All_,simple_]
#treatments_dic=dict(saline_,PAR4_,MIPS_,ASA_,SQ_,CMFDA_,All_,simple_)
shorttolong_dic = dict(zip(shortnames_, longnames_))
longtoshort_dic = dict(zip(longnames_,shortnames_))


#---------------------------------------------------------------------------
#Lists of variables for certain applications
#----------------------------------------------------------------------
xtra_vars_=['position','inside_injury','injury_zone','height','z_pos','phase','minute']
old_xtra_vars_=['position','inside_injury','height','z_pos']
new_xtra_vars_=['injury_zone','phase','minute','quadrant','quadrant1']
thr_reg_vars=['position','inside_injury','injury_zone','quadrant','quadrant1']
time_vars=['sec','frame','min','phase']

#---------------------------------------------------------------------------
# Lists specifying orders of variables
#---------------------------------------------------------------------------
bol_order=[True,False]
mov_class_order=['still','contractile','loose','none']
mov_class_order1=['still','contractile','loose']
movement_order=['immobile','contracting','drifting','unstable','none']
movement_order1=['immobile','contracting','drifting','unstable']
movements_order2=['immobile','contracting','drifting']
position_order=['head','tail','outside']
height_order=['bottom','middle','top']
phase_order=['Early','Mid','Late']
quadrant_order=['anterior','lateral','posterior']
quadrant1_order=['core','anterior','lateral','posterior']

var_order=dict(inside_injury=bol_order, mov_class=mov_class_order1, 
               movement=movements_order2, injury_zone=bol_order, 
               position=position_order, height=height_order, phase=phase_order, 
               quadrant=quadrant_order,quadrant1=quadrant1_order)       


    
def settings():
    # Settings and Parameters
    #----------------------------------------------------------------------
    warnings.simplefilter(action = "ignore", category = FutureWarning)

    pd.set_option('display.max_rows', 300)
    pd.set_option('display.max_columns', 300)
    #pd.option_context('display.max_rows', None, 'display.max_columns', None)
    
    #AESTHETICS
    #Parameters for image export 
    #---------------------------------------------------------------------------
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['image.cmap'] = 'viridis'#'turbo'#'viridis' 'magma'
    
    #Plot styles 
    #---------------------------------------------------------------------------
    #sns.set_style('darkgrid')
    #sns.set_style('whitegrid')
    sns.set_style('ticks')
    #sns.set_context("talk") #paper,notebook,talk,poster
    sns.set_context("talk")
    #plt.rcParams['image.cmap'] = 'jet'
    plt.rcParams['image.interpolation'] = 'none'
    # CHOOSE COLOR PALETTE 
    # ------------------------------------------------------------------------
    #sns.set_palette('Dark2')# palette='tab20'sns.set_palette('Paired')
    sns.set_palette('Set1')#Set3