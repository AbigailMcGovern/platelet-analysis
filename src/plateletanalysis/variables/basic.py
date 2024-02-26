import numpy as np
import pandas as pd
import math as m
from toolz import curry

#from plateletanalysis.variables.measure import quantile_normalise_variables, quantile_normalise_variables_frame
#from .. import config as cfg
#from .transform import spherical_coordinates
from tqdm import tqdm

# --------------------------
# Add Variables to DataFrame
# --------------------------


def add_basic_variables_to_files(file_paths, stab=True, density=False, nba=False, cont=False):
    data = []
    for p in file_paths:
        df = pd.read_parquet(p)
        df['treatment'] = df['path'].apply(get_treatment_name)
        df.to_parquet(p)
        if 'nrtracks' not in df.columns.values:
            df = add_nrtracks(df)
            df.to_parquet(p)
        if 'tracknr' not in df.columns.values:
            df = add_tracknr(df)
            df.to_parquet(p)
        if 'time (s)' not in df.columns.values:
            df = add_time_seconds(df)
            df.to_parquet(p)
        if 'terminating' not in df.columns.values:
            df = add_terminating(df)
            df.to_parquet(p)
        if 'sliding (ums^-1)' not in df.columns.values:
            df = add_sliding_variable(df)
            df.to_parquet(p)
        if 'minute' not in df.columns.values:
            df = time_minutes(df)
            df.to_parquet(p)
        if 'total time tracked (s)' not in df.columns.values:
            df = time_tracked_var(df)
            df.to_parquet(p)
        if 'tracking time (s)' not in df.columns.values:
            df = tracking_time_var(df)
            df.to_parquet(p)
        if 'stab' not in df.columns.values and stab:
            from plateletanalysis.variables.measure import stability
            df = stability(df)
            df.to_parquet(p)
        if 'size' not in df.columns.values:
            df = size_var(df)
            df.to_parquet(p)
        if 'inside_injury' not in df.columns.values:
            df = inside_injury_var(df)
            df.to_parquet(p)
        if 'nba_5' not in df.columns.values and nba:
            from plateletanalysis.variables.neighbours import average_neighbour_distance
            df = average_neighbour_distance(df) # check
            df.to_parquet(p)
        if 'nb_density_15' not in df.columns.values and density:
            from plateletanalysis.variables.neighbours import add_neighbour_lists, local_density
            if 'nb_particle_15' not in df.columns.values:
                df = add_neighbour_lists(df)
            df = local_density(df)
            df.to_parquet(p)
        if 'cont' not in df.columns.values and cont:
            from plateletanalysis import contraction
            df = contraction(df)
            df.to_parquet(p)
        if 'phi' not in df.columns.values:
            from plateletanalysis.variables.transform import spherical_coordinates
            df = spherical_coordinates(df)
            df.to_parquet(p)
        data.append(df)
    df = pd.concat(data).reset_index(drop=True)
    del data
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
    df.loc[:,'minute'] = pd.cut(df['time (s)'], 10, labels=np.arange(1,11,1))
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


def tracknr_variable(pc, sample_col='path'):
    try:
        pc = pc.drop(['level_0'], axis=1)
    except:
        pass
    pc = pc.reset_index()
    # Tracknr raknar vilken trackning i ordningen en viss observation ar
    tracknr=pc.groupby([sample_col,'particle'])['frame'].rank()
    pc['tracknr']=tracknr
    return pc


def tracked_variable(df):
    df['tracked'] = df['nrtracks'] > 1
    return df



def get_treatment_name(inh): # need to rename from last run 
    if 'saline' in inh or 'Saline' in inh:
        out = 'saline'
    elif 'cang' in inh or 'Cang' in inh:
        out = 'cangrelor'
    elif 'veh-mips' in inh:
        out = 'DMSO (MIPS)'
    elif 'mips' in inh or 'MIPS' in inh or 'Injury 2-4 (MIPS effect)' in inh:
        out = 'MIPS'
    elif 'sq' in inh:
        out = 'SQ'
    elif 'par4--biva' in inh:
        out = 'PAR4-- bivalirudin'
    elif 'par4--' in inh:
        out = 'PAR4--'
    elif 'biva' in inh or 'Bivalirudin' in inh:
        out = 'bivalirudin'
    elif 'SalgavDMSO' in inh or 'gavsalDMSO' in inh or 'galsavDMSO' in inh:
        out = 'DMSO (salgav)'
    elif 'Salgav' in inh or 'gavsal' in inh:
        out = 'salgav'
    elif 'DMSO' in inh or 'DMSO 20ul' in inh:
        out = 'DMSO (MIPS)'
    elif 'dmso' in inh:
        out = 'DMSO (SQ)'
    elif 'ctrl' in inh or 'Ctrl' in inh:
        out = 'control'
    else:
        out = inh
    return out


def fsec_var(df, n_hues = 10): #if you use 10, 60s each (10 intervals)
    '''bin into intervals dep on when pl was first detected'''
    if 'fframe' not in df.columns:
        df = fframe_var(df)
    
    edges = np.arange(0,205,10)#np.arange(0,196,3)
    labels = list(np.linspace(0,570,20).astype('int'))#np.round(np.arange(0,600,9.3),0).astype('int')

    df.loc[:,'fsec'] = pd.cut(df['fframe'], edges, labels= labels, include_lowest= True)
    return df


def fframe_var(df):
    '''first frame'''
    first_frame = df.sort_values(by = ['path','particle','tracknr']).groupby(['path','particle'])['frame'].nth(0).rename('fframe')
    df = df.merge(first_frame, on = ['path','particle'])
    return df


def add_region_category(df):
    rcyl = (df.x_s ** 2 + df.ys ** 2) ** 0.5
    df['rcyl'] = rcyl
    df['region'] = [None, ] * len(df)
    # center
    sdf = df[df['rcyl'] <= 37.5]
    idxs = sdf.index.values
    df.loc[idxs, 'region'] = 'center'
    # outer regions
    # 45 degrees = 0.785398
    sdf = df[df['rcyl'] > 37.5]
    # anterior
    rdf = sdf[sdf['phi'] > 0.785398]
    idxs = rdf.index.values
    df.loc[idxs, 'region'] = 'anterior'
    # lateral
    rdf = sdf[(sdf['phi'] < 0.785398) & (sdf['phi'] > -0.785398)]
    idxs = rdf.index.values
    df.loc[idxs, 'region'] = 'lateral'
    # posterior
    rdf = sdf[sdf['phi'] < -0.785398]
    idxs = rdf.index.values
    df.loc[idxs, 'region'] = 'posterior'
    return df


def add_quadrant(df):
    rcyl = (df.x_s ** 2 + df.ys ** 2) ** 0.5
    df['rcyl'] = rcyl
    df['quadrant'] = [None, ] * len(df)
    # anterior
    rdf = df[df['phi'] > 0.785398]
    idxs = rdf.index.values
    df.loc[idxs, 'quadrant'] = 'anterior'
    # lateral
    rdf = df[(df['phi'] < 0.785398) & (df['phi'] > -0.785398)]
    idxs = rdf.index.values
    df.loc[idxs, 'quadrant'] = 'lateral'
    # posterior
    rdf = df[df['phi'] < -0.785398]
    idxs = rdf.index.values
    df.loc[idxs, 'quadrant'] = 'posterior'
    return df


def add_nrtracks(df):
    for k, g in df.groupby(['path', 'particle', ]):
        n = len(g)
        idxs = g.index.values
        df.loc[idxs, 'nrtracks'] = n
    return df


def add_tracknr(df):
    df = df.sort_values('frame')
    for k, g in df.groupby(['path', 'particle', ]):
        track_nr = range(1, len(g) + 1)
        idxs = g.index.values
        df.loc[idxs, 'tracknr'] = track_nr
    return df


def add_phase(df, phases={'growth' : (0, 260), 'consolidation' : (260, 600)}):
    df['phase'] = [None, ] * len(df)
    for phase in phases:
        sdf = df[(df['time (s)'] > phases[phase][0]) & (df['time (s)'] > phases[phase][1])]
        idxs = sdf.index.values
        df.loc[idxs, 'phase'] = phase
    return df



def add_time_seconds(df, frame_col='frame'):
    df['time (s)'] = df[frame_col] / 0.321764322705706
    return df


def add_sliding_variable(df):
    #df['sliding (ums^-1)'] = [None, ] * len(df)
    # not moving in direction of blood flow
    sdf = df[df['dvy'] >= 0]
    idxs = sdf.index.values
    df.loc[idxs, 'sliding (ums^-1)'] = 0
    # moving in the direction of blood flow
    sdf = df[df['dvy'] < 0]
    new = np.where(df['dvy'].values < 0, np.abs(df['dvy'].values), 0)
    df['sliding (ums^-1)'] = new
    #idxs = sdf.index.values
    #new = np.abs(sdf['dvy'].values)
    #df.loc[idxs, 'sliding (ums^-1)'] = new
    return df


def tracking_time_var(df):
    df['tracking time (s)'] = df['tracknr'] / 0.321764322705706
    return df


def time_tracked_var(df):
    df['total time tracked (s)'] = df['nrtracks'] / 0.321764322705706
    return df


def add_terminating(df):
    df['terminating'] = [False, ] * len(df)
    for k, g in df.groupby(['path', ]):
        t_max = g['frame'].max()
        sdf = g[g['frame'] != t_max]
        term = sdf['nrtracks'] == sdf['tracknr']
        idxs = sdf.index.values
        df.loc[idxs, 'terminating'] = term
    return df


def add_normalised_ca_pcnt(df):
    for k, g in df.groupby(['path', 'frame']):
        ca_max = g['ca_corr'].max()
        ca_norm = g['ca_corr'] / ca_max * 100
        idxs = g.index.values
        df.loc[idxs, 'Ca2+ pcnt max'] = ca_norm
    return df


def add_shedding(df):
    df['shedding'] = [False, ] * len(df)
    nits = 0
    for k, g in df.groupby(['path', 'particle']):
        nits += 1
    with tqdm(total=nits) as progress:
        for k, g in df.groupby(['path', 'particle']):
            #shed = df['terminating'].sum() # was going to take 11 hours
            if True in df['terminating'].values:
                idxs = g.index.values
                df.loc[idxs, 'shedding'] = True
            progress.update(1)
    return df


def size_var(df):
    '''Was this from a thrombus that is from the largest 50th centile or the smallest?'''
    df_size = df.groupby(['treatment', 'path'])['particle'].apply(_count).reset_index(drop=False)
    df = df.set_index('path')
    print(df.index.values[:10])
    for k, grp in df_size.groupby('treatment'):
        idxs = grp['path'].values
        #print(idx)
        labs = pd.Series(pd.qcut(grp['particle'].values, 2, labels=['small', 'large']))
        for i, idx in enumerate(idxs):
            df.loc[idx, 'size'] = labs.values[i]
    df = df.reset_index(drop=False)
    return df 


def _count(df):
    return len(pd.unique(df))

@curry
def _map_size(dict, df):
    pass

def hsec_var(df):
    '''which 100 seconds?'''
    t = np.arange(0,700,100)
    t_i = [str(t1)+'-'+ str(t2) for t1,t2 in zip(t[:-1],t[1:])]
    df.loc[:,'hsec'] = pd.cut(df['time (s)'], t, labels=t_i, include_lowest= True)
    
    return df


def inside_injury_var(df): 
    df=dist_c_var(df)
    df['inside_injury'] = df.dist_c < 37.5
    return df

    
def dist_c_var(df):# Creates variables dist_c & dist_cz that give distance from center
    df['dist_c']=((df.loc[:,'x_s'])**2+(df.loc[:,'ys'])**2)**0.5
    df['dist_cz']=((df.loc[:,'x_s'])**2+(df.loc[:,'ys'])**2+(df.loc[:,'zs'])**2)**0.5
    return df


def quadrant_var(df):
    df['quadrant']='lateral'
    df.loc[(df['ys']>df['x_s'])&(df['ys']>-df['x_s']),'quadrant']='anterior'
    df.loc[(df['ys']<df['x_s'])&(df['ys']<-df['x_s']),'quadrant']='posterior'
    return df


def rename_channel_vars(df):
    rename = {
        'Alxa 647: mean_intensity' : 'c2_mean', 
        'Alxa 647: max_intensity' : 'c2_max', 
        'GaAsP Alexa 488: mean_intensity' : 'c0_mean',
       'GaAsP Alexa 488: max_intensity' : 'c0_max',
       'GaAsP Alexa 568: mean_intensity' : 'c1_mean',
       'GaAsP Alexa 568: max_intensity': 'c1_max'
    }
    df = df.rename(columns=rename)
    return df


def tri_phase_var(df):
    if 'time (s)' not in df.columns.values:
        df = time_seconds(df)
    u_bins = [100, 300, 600]
    l_bins = [0, 100, 300]
    out_type = 'string'
    bin_func = _value_bin(u_bins, l_bins, out_type)  
    df['tri_fsec'] = df['time (s)'].apply(bin_func)  
    return df



def bin_by_var_linear(df, var, ub, lb, n_bins, bin_name, out_type='string'):
    u_bins = np.linspace(lb, ub, n_bins)[:-1]
    l_bins = np.linspace(lb, ub, n_bins)[1:]
    bin_func = _value_bin(u_bins, l_bins, out_type)  
    vals = df[var].apply(bin_func)  
    df[bin_name] = vals


@curry
def _value_bin(u_bins, l_bins, out_type, val):
    for lb, ub in zip(u_bins, l_bins):
        if val >= lb and val < ub:
            if out_type == 'mean':
                b = (lb + ub) / 2
            elif out_type == 'string':
                b = f'{lb}-{ub}'
            return b
        

def isoA_var(df, nA_in_injury=10):
    if 'dist_c' not in df.columns:
        df = dist_c_var(df)
    inj_zone_A = np.pi * (37.5 ** 2) / 2
    A_step = inj_zone_A/nA_in_injury #CHANGED FROM 3
    A_ = np.arange(0,A_step*(nA_in_injury*10+1),A_step)
    radii = ((2*A_/(np.pi)))**(1/2)
    radii[-1] = 250
    df['iso_A'] = pd.cut(df['dist_c'],radii,labels=radii[1:]).astype('float64').round(1)
    return df


def classify_exp_type(path):
    if path.find('exp5') != -1:
        return '10-20 min'
    elif path.find('exp3') != -1:
        return '0-10 min'
    else:
        return 'other'
    

def time_bin_1_2_5_10(t):
    lbs = [0, 60, 120, 300]#, 600]
    ubs = [60, 120, 300, 600]#, 1200]
    for l, u in zip(lbs, ubs):
        if t >= l and t < u:
            return f'{l}-{u} s'
        
def time_bin_30_3060_1_2_5_10(t):
    lbs = [0, 30, 60, 120, 300]#, 600]
    ubs = [30, 60, 120, 300, 600]#, 1200]
    for l, u in zip(lbs, ubs):
        if t >= l and t < u:
            return f'{l}-{u} s'
        

def time_bin_2_5_10_20(t):
    lbs = [0, 120, 300, 600]
    ubs = [120, 300, 600, 1200]
    for l, u in zip(lbs, ubs):
        if t >= l and t < u:
            return f'{l}-{u} s'
        
        
def cyl_bin(t):
    lbs = np.linspace(0, 100)[0:-1]
    ubs = np.linspace(0, 100)[1:]
    for l, u in zip(lbs, ubs):
        if t >= l and t < u:
            return l + 0.5 * (u - l)
        

@curry
def curried_timebin(lbs, ubs, t):
    for l, u in zip(lbs, ubs):
        if t >= l and t < u:
            return f'{l}-{u} s'
        

@curry
def curried_midpoint_bin(lbs, ubs, t):
    for l, u in zip(lbs, ubs):
        if t >= l and t < u:
            return l + 0.5 * (u - l)
        

def adjust_time_for_exp_type(df):
    if 'time (s)' not in df.columns:
        df = time_seconds(df)
    if 'exp_type' not in df.columns:
        df['exp_type'] = df['path'].apply(classify_exp_type)
    for k, grp in df.groupby('exp_type'):
        if k == '10-20 min':
            idxs = grp.index.values
            df.loc[idxs, 'time (s)'] = df.loc[idxs, 'time (s)'] + 600
    return df


def add_psel_bin(vals):
    if vals.mean() > 544:
        return [True, ] * len(vals)
    else:
        return [False, ] * len(vals)
    

def psel_bin(df):
    print('psel bin')
    for k, grp in df.groupby(['path', 'particle']):
        idx = grp.index.values
        vals = add_psel_bin(grp['p-sel average intensity'].values)
        df.loc[idx, 'psel'] = vals
    #df['psel'] = df.groupby(['path', 'particle'])['p-sel average intensity'].apply(add_psel_bin)
    return df





