import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy.stats import pearsonr
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import percentileofscore

# --------------------
# Correlation analyses
# --------------------

def cross_corr(df, cols, save):
    n_iter = len(cols)
    results = {
        'variables' : cols
    }
    heatmap = {
        'variables' : cols
    }
    with tqdm(total=n_iter) as progress:
        for i, col in enumerate(cols):
            n_r = f'{col}_r'
            n_p = f'{col}_p'
            rs = []
            ps = []
            for c in cols:
                r, p = pearsonr(df[col].values, df[c].values)
                rs.append(r)
                ps.append(p)
            results[n_r] = rs
            results[n_p] = ps
            heatmap[col] = rs
            progress.update(1)
    results = pd.DataFrame(results)
    results.to_csv(save)
    heatmap = pd.DataFrame(heatmap)
    heatmap = heatmap.set_index('variables')
    matrix = np.triu(heatmap.values)
    ax = sns.heatmap(heatmap, annot=True,  linewidths=.5, cmap="vlag", vmin=-1, vmax=1, mask=matrix)
    plt.show()


# -----------------------
# Multivariate Regression
# -----------------------

def multivariate_regression_model(df, formula='dv ~ nb_ca_corr_15 + fibrin_dist_pcnt'):
    est = smf.ols(formula=formula, data=df).fit()
    print(est.summary())



# ---------------------------------------------
# Spatial pairwise condition-control comparison
# ---------------------------------------------

def spherical_coordinate_bins(
        df, 
        rho_pcnt_bins=(50, 75, 90, 98), 
        phi_pcnt_bins=(25, 50, 75, 100), 
        theta_pcnt_bins=(25, 50, 75, 100)
        ):
    '''
    Pairwise t tests using bonferroni or false discovery correction to correct for multiple comparisons. 
    In line with approaches used for brain mri/fmri/dti analyses, but with far fewer comparisons. 
    Clot is broken into segments along anterior-posterior angle, top-bottom angle, and by distance from centre.
    The 
    '''
    files = pd.unique(df['path'])
    for f in files:
        fdf = df[df['path'] == f]
        # asymetry of the clot is determined by phi so binning will be done separately for each percentile of phi
        for phi_pcnt in phi_pcnt_bins:
            phi_df = fdf[fdf['phi_pcnt'] < phi_pcnt]
            idxs = phi_df.index.values
            df.loc[idxs, 'ptr_pcnt_bin'] = f'({phi_pcnt}, '
            # find the percentiles for theta and rho for this quadrant
            rho = phi_df['rho'].values
            rho_pcnt = np.array([percentileofscore(rho, d) for d in rho])
            theta = phi_df['theta'].values
            rho_pcnt = np.array([percentileofscore(theta, d) for d in theta])
            for bin in theta_pcnt_bins:
                tidxs = idxs[np.where(theta < bin)]
                df.loc[ridxs, 'ptr_pcnt_bin'] = df.loc[tidxs, 'ptr_pcnt_bin'] + f'{bin})'
            for bin in rho_pcnt_bins:
                ridxs = idxs[np.where(rho < bin)]
                df.loc[ridxs, 'ptr_pcnt_bin'] = df.loc[ridxs, 'ptr_pcnt_bin'] + f'{bin})'


                


def spherical_coordinate_bins_comparison(df, treatment_col='inh', inh_ctrl=('_saline_', '_cang_')):
    pass