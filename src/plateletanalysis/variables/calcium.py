from scipy.signal import find_peaks
import numpy as np
import pandas as pd
#from ....old_modules import config as cfg

# Global parameters for calcium analysis
#----------------------------------------------------------------------

calc_var='ca_corr'
#ca_measure=max_peak_prom
#calc_var='c0_mean'

# New calcium variables
#----------------------------------------------------------------------
def corrected_calcium(df, by_tx=False):
    if not by_tx:
        df['ca_corr'] = calc_comp(df)
    else:
        for k, grp in df.groupby('treatment'):
            idxs = grp.index.values
            vals = calc_comp(grp)
            df.loc[idxs, 'ca_corr'] = vals
    return df


def calc_comp(df):
    z  = df.zs
    c = df.c0_mean
    #try:
     #   d_max = pd.read_csv('df_reg zz 98.csv').to_dict('records')[0]#'Calcium comp\\df_reg zz max_calc.csv'
      #  d_min = pd.read_csv('df_reg zz 1.csv').to_dict('records')[0]#'Calcium comp\\df_reg zz min_calc.csv'
    #except:
    d_max = {'a':814.697988,'b':-22.386315,'c':0.196006}
    d_min = {'a':159.465109,'b':-1.330306,'c':0.011518}  
    c_high = d_max['a'] + d_max['b'] * z + d_max['c'] * (z**2)
    c_low = d_min['a'] + d_min['b'] * z + d_min['c'] * (z**2)
    c_corr = 100 * (c - c_low) / (c_high - c_low)
    c_corr = c_corr.clip(0)
    return c_corr

# Functions for analysis of calcium in individual platelets
#----------------------------------------------------------------------

def calc_maxmin(df):
    delta_ca=df[calc_var].max()-df[calc_var].min()
    return delta_ca

def count_peaks(x):
    peaks, _ = find_peaks(x, height=100,threshold=10,prominence=50)#
    nr_peaks=len(peaks)
    return nr_peaks

def sum_peaks(x):
    peaks, props = find_peaks(x, height=100,threshold=10,prominence=50)
    sum_peaks=np.sum(props['left_thresholds'])+np.sum(props['left_thresholds'])
    return sum_peaks

def max_peak(x):
    peaks, props = find_peaks(x, height=200,threshold=10)
    if len(peaks)>0:
        peak_diff=props['left_thresholds']+props['right_thresholds']
        peak_ind=np.argsort(peak_diff)
        max_peaks=peak_diff[peak_ind]
        max_peak=np.sum(max_peaks[-3:])
    else:
        max_peak=0
    return max_peak

def max_peak_prom(x):
    peaks, props = find_peaks(x, height=150,prominence=70)#height=200
    if len(peaks)>0:
        try:
            peak_diff=props['prominences']
            peak_ind=np.argsort(peak_diff)
            max_peaks=peak_diff[peak_ind]
            max_peak=np.max(max_peaks)
        except (RuntimeError, TypeError, KeyError):
            print(props)
    else:
        max_peak=0
    return max_peak

def stat_peaks(dfg):
    peaks, props = find_peaks(dfg, height=200,prominence=70)
    nr_peaks=len(peaks)
    peak_auc=np.sum(props['prominences'])
    stat_dic={'nr_peaks':nr_peaks,'peak_auc':peak_auc}
    return stat_dic

def nr_peaks(dfg):
    peaks, props = find_peaks(dfg[calc_var], height=200,prominence=70)
    nr_peaks=len(peaks)
    peak_auc=np.sum(props['prominences'])
    dfg['nr_peaks']=nr_peaks
    dfg['peak_auc']=peak_auc
    dfg['']
    return dfg

# Aggregate functions for analysis of calcium data
#----------------------------------------------------------------------
def df_plt_ca(df,sort_var,exp_nr):#Before sel_peak_prom
    y_groups=np.sort(df[sort_var].unique())#[0:4]#['30ADP','P4'])
    peaklist_top=[]
    ca_measure=max_peak_prom #Select function for summary of calcium variable, available: max_peak,sum_peaks,max_peak_prom
    for level in y_groups:
        grouped=df[((df.nrtracks>10)&(df[sort_var]==level))].groupby(['exp_id','particle',sort_var])
        dfi=grouped[calc_var].agg(ca_measure).reset_index()#sum_peaks
        if len(dfi)>0:
            dfi.columns=['exp_id','particle',sort_var,'Sum peaks']
            peak_list=dfi.sort_values(by='Sum peaks',ascending=False).reset_index()
            peaklist_top.append(peak_list.iloc[0:exp_nr])
    df_peaklist=pd.concat(peaklist_top,axis=0).reset_index().drop(['index'],axis=1)
    df_peaklist=df_peaklist.rename(columns={'level_0':'delta_pos'})
    return df_peaklist

def select_plts_ca(df,sort_var,exp_nr):
    y_groups=np.sort(df[sort_var].unique())#[0:4]#['30ADP','P4'])
    ca_delta_top=[]
    for level in y_groups:
        print('zlevel',level)
        grouped=df[((df.nrtracks>20)&(df[sort_var]==level))].groupby(['exp_id','particle'])
        ca_delta=grouped.apply(calc_maxmin).reset_index()
        if len(ca_delta)>0:
            print(ca_delta.head(5))
            ca_delta.columns=['exp_id','particle','delta']
            ca_delta[sort_var]=level
            ca_delta2=grouped.sum()[calc_var].reset_index()
            ca_delta[calc_var]=ca_delta2[calc_var]
            ca_delta['combi']=ca_delta.delta*ca_delta[calc_var]
            ca_delta=ca_delta.sort_values(by='combi',ascending=False).reset_index()
            ca_delta_top.append(ca_delta.iloc[0:exp_nr])
    df_ca_delta=pd.concat(ca_delta_top,axis=0).reset_index().drop(['index'],axis=1)
    return df_ca_delta

def df_top_ca(df,df_list,sort_var):
    dftop_list=[]
    for row_nr in range(len(df_list)):
        row=df_list.iloc[row_nr].copy()
        df_rows=df[((df.particle==row.particle)&(df.exp_id==row.exp_id))].copy()#(df[sort_var]==row[sort_var])&
        df_rows['delta_pos']=row.delta_pos.item()
        dftop_list.append(df_rows) 
    dftop=pd.concat(dftop_list)
    return dftop

def calc_stat_df(df,measure,nr_mean):
    dfg=df.groupby(['zled'])[measure]
    count=dfg.count()
    mean=dfg.mean()
    mean_low=dfg.nsmallest(nr_mean).mean(level=0)
    mean_high=dfg.nlargest(nr_mean).mean(level=0)#.name('mean_high')#.rename(columns={measure:'mean high'})
    first_perc=dfg.quantile(.01)
    last_perc=dfg.quantile(.99)
    df_stat=pd.concat([count, mean,first_perc,mean_low,last_perc,mean_high],axis=1).reset_index()#.rename(columns=['mean high','mean low'])
    df_stat=df_stat.set_axis(['zled','No observations','Mean','First perc','Mean 1000 lowest', 'Last perc','Mean 1000 highest'], axis=1, inplace=False)
    return df_stat