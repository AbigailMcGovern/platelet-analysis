import numpy as np
import pandas as pd
import math as m

from plateletanalysis.variables.measure import quantile_normalise_variables, quantile_normalise_variables_frame
from .. import config as cfg
from .transform import spherical_coordinates

# --------------------------
# Add Variables to DataFrame
# --------------------------

def add_basic_variables(df):
    '''
    Adds the following variables:
    - z_led
    - dist_c
    - dist_cz
    - time
    - minute
    - injury_zone
    - height
    - z_pos
    - zz
    - position
    - inside_injury
    - tracked
    - mov_class
    - movement
    - exp_id
    - inh_exp_id
    - tracknr
    '''
    df = led_bins_var(df)
    df = dist_c_var(df)
    df = time_var(df)
    df = minute_var(df)
    df = injury_zone_var(df)
    df = height_var(df)
    df = z_pos_var(df)
    df = zz_var(df)
    df = position_var(df)
    df = inside_injury_var(df)
    df = tracked_variable(df)
    df = add_tot_variables(df)
    df = new_exp_ids(df)
    df = mov_class_var(df)
    df = movement_var(df)
    df = tracknr_variable(df)
    return df



# ----------------------------------------
# Individual Variable Generating Functions
# ----------------------------------------
# By Niklas - originally in data_func.py

def led_bins_var(
    dfg, 
    z_col='zs',   
    t_col='frame', 
    ):
    '''
    Create binned variables for z position (zled) and time in seconds (tled).
    Z is binned from 0 - 68 with 17 bins. 
    T is only computed if sec is in the dataframe and is binned from 0-600
    seconds with 97 bins.
    
    '''#Creates bin variables tled & zled  for heatmaps etc. 
    zbins=[-2,4]+np.arange(8,72,4).tolist()#np.arange(-8,72,4)
    zgroup_names=np.round(np.linspace(0,68,17),0)#np.round(np.linspace(-6,74,19),0)
    #tbins=np.arange(0,196,2)
    tbins=np.linspace(0,194,98)#np.linspace(0,192,97)
    tgroup_names=np.round(np.linspace(0,600,97),0)
    #tgroup_names=np.arange(0,196,2)
    if 'zs' in dfg.columns:  
        dfg['zled'] = pd.cut(dfg[z_col], zbins, labels=zgroup_names,include_lowest=True).astype('Int64')
    if 'sec' in dfg.columns:
        dfg['tled'] = pd.cut(dfg[t_col], tbins, labels=tgroup_names,include_lowest=True).astype('Int64')    
    return dfg


def dist_c_var(df):# Creates variables dist_c & dist_cz that give distance from center
    '''
    Compute variables giving total distance from the centre (dist_c) and distance from
    the centre in the z axis (dist_cz). 
    '''
    df['dist_c']=((df.loc[:,'x_s'])**2+(df.loc[:,'ys'])**2)**0.5
    df['dist_cz']=((df.loc[:,'x_s'])**2+(df.loc[:,'ys'])**2+(df.loc[:,'zs'])**2)**0.5
    return df


def isovol_bin_var(df): # Isnt in the dataframes I've seen (check use)
    if 'dist_cz' in df.columns:
        inj_zone_vol=(2/3)*m.pi*(37.5**3)
        vol_step=inj_zone_vol/10
        volumes_=np.arange(0,vol_step*201,vol_step)
        radii=((3*volumes_/(2*m.pi)))**(1/3)
        radii[-1]=250
        df['iso_vol']=pd.cut(df['dist_cz'],radii,labels=radii[1:])
    return df


def time_var(df):# Creates time variables time, minute and phase from frame
    df['time']=df['frame']*3.1
    return df


def minute_var(df):
    if 'time' not in df.columns:
        df=time_var(df)
    df.loc[:,'minute'] = pd.cut(df['time'], 10, labels=np.arange(1,11,1))
    return df


def time_seconds(df):
    df['time (s)'] = df['frame'] / 0.321764322705706
    return df


def time_minutes(df):
    if 'time (s)' not in df.columns:
        df['time (s)'] = df['frame'] / 0.321764322705706
    df.loc[:,'minute'] = pd.cut(df['time'], 10, labels=np.arange(1,11,1))
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


def inside_injury_var(df): 
    df['inside_injury']=df.position.isin(['head'])
    return df


def mov_class_var(df):#New definition 191209
    for exp_id in pd.unique(df.exp_id):
        dfi=df[df.exp_id==exp_id].copy()
        try:
            still = pd.unique(dfi[((dfi.displ_tot/dfi.nrtracks)<0.1) & (dfi.displ_tot<4)]['particle'])
            loose = pd.unique(dfi[(dfi.displ_tot>5) & ((dfi.cont_tot/dfi.displ_tot)<0.2)]['particle'])
            contractile = pd.unique(dfi[((dfi.cont_tot/dfi.displ_tot)>0.5) & (dfi.displ_tot>1)]['particle'])
        except TypeError:
            print(exp_id,dfi.displ_tot.dtypes,dfi.nrtracks.dtypes,(dfi.displ_tot/dfi.nrtracks))
        df.loc[(df.exp_id==exp_id) & (df['particle'].isin(still)),'mov_class']="still"
        df.loc[(df.exp_id==exp_id) & (df['particle'].isin(loose)),'mov_class']="loose"
        df.loc[(df.exp_id==exp_id) & (df['particle'].isin(contractile)),'mov_class']="contractile"
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
    longnames_list=[
        'Bivalirudin',
        'Cangrelor',
        'CMFDA',
        'Control',
        'MIPS',
        'Saline',
        'SQ',
        'Vehicle MIPS',
        'Vehicle SQ',
        'PAR4+/-',
        'PAR4-/+',
        'PAR4-/- + biva',
        'PAR4-/-',
        'ASA + Vehicle',
        'ASA','Salgav + Vehicle',
        'Salgav'
        ]
    inh_dic={
        'vehicle MIPS':'vehicle\nMIPS', 
        'salgavDMSO':'salgav\nDMSO', 
        'vehicle sq':'vehicle\nsq', 
        'par4--biva':'par4--\nbiva'}


def add_tot_variables(pc1):
    #Satter index pa path och particle for att kunna konkaternera dataframe senare
    particle = pc1['particle']
    path = pc1['path']
    pc1=pc1.reset_index().set_index(['path','particle'])
    #Grupperar data pa path och partikel for att kunna analysera olika partiklar for sig
    grouped=pc1.groupby(['path','particle'])
    #Raknar antalet observationer for varje unik partikel
    counted=grouped.count()
    #Variabeln nrtracks kollar hur manga tracks en viss partikel ar trackad
    #pc1['nrtracks']=counted.frame
    #Adderar de olika variablerna och beraknar summan av alla observationer
    summed=grouped.sum()
    #cont_tot summerar den totala kontraktionen for en partikel
    pc1['cont_tot']=summed.cont
    # displ_tot summerar den totala forflyttningen for en partikel
    pc1['displ_tot']=abs(summed.dvx)+abs(summed.dvy)
    # dvz_tot summerar den totala forflyttningen i z-led for en partikel
    pc1['dvz_tot']=summed.dvz
    # dvz_tot summerar den totala forflyttningen i y-led for en partikel
    pc1['dvy_tot']=summed.dvy
    pc1 = pc1.reset_index()
    pc1['particle'] = particle
    pc1['path'] = path
    return pc1


def tracknr_variable(pc):
    try:
        pc = pc.drop(['level_0'], axis=1)
    except:
        pass
    pc = pc.reset_index()
    # Tracknr raknar vilken trackning i ordningen en viss observation ar
    tracknr=pc.groupby(['path','particle'])['frame'].rank()
    pc['tracknr']=tracknr
    return pc


def tracked_variable(df):
    df['tracked'] = df['nrtracks'] > 1
    return df



def get_treatment_name(inh): # need to rename from last run 
    if 'saline' in inh:
        out = 'saline'
    elif 'cang' in inh:
        out = 'cangrelor'
    elif 'veh-mips' in inh:
        out = 'MIPS vehicle'
    elif 'mips' in inh or 'MIPS' in inh:
        out = 'MIPS'
    elif 'sq' in inh:
        out = 'SQ'
    elif 'par4--biva' in inh:
        out = 'PAR4-- bivalirudin'
    elif 'par4--' in inh:
        out = 'PAR4--'
    elif 'biva' in inh:
        out = 'bivalirudin'
    elif 'SalgavDMSO' in inh or 'gavsalDMSO' in inh or 'galsavDMSO' in inh:
        out = 'DMSO (salgav)'
    elif 'Salgav' in inh or 'gavsal' in inh:
        out = 'salgav'
    elif 'DMSO' in inh:
        out = 'DMSO (MIPS)'
    elif 'dmso' in inh:
        out = 'DMSO (SQ)'
    elif 'ctrl' in inh:
        out = 'control'
    else:
        out = inh
    return out