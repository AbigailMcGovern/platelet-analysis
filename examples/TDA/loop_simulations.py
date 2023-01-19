from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.stats import scoreatpercentile
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from plateletanalysis.variables.measure import quantile_normalise_variables, quantile_normalise_variables_frame



def view_scatter_matrix(func, sigmas, sizes):
    fig, axs = plt.subplots(len(sigmas), len(sizes), sharex=True, sharey=True)
    for i in range(len(sigmas)):
        for j in range(len(sizes)):
            data = func(sigmas[i], sizes[j])
            x, y = data[:, 0], data[:, 1]
            ax = axs[i, j]
            ax.scatter(x, y, s=1)
            ax.set_title(f'sigma {sigmas[i]}, size {sizes[j]}')
    plt.show()



def simulate_disc(sigma, size, r=100):
    sigma = r * sigma
    x = np.random.normal(scale=sigma, size=size)
    y = np.random.normal(scale=sigma, size=size)
    #y_pcnt = np.array([percentileofscore(y, v) for v in y])
    #x_pcnt = np.array([percentileofscore(x, v) for v in x])
    result = np.stack([x, y]).T
    return result



def simulate_1_loop(sigma, size, r=25):
    result = _loop_coords(size, sigma, r)
    return result



def _loop_coords(n, s, r):
    s = r * s
    theta = np.linspace( 0 , 2 * np.pi , n)
    radius = r
    x = radius * np.cos( theta )
    y = radius * np.sin( theta )
    noise_x = np.random.normal(scale=s, size=n)
    noise_y = np.random.normal(scale=s, size=n)
    x = x + noise_x
    y = y + noise_y
    result = np.stack([x, y]).T
    result = result + 50
    return result


def simulate_2_loop(sigma, size):
    res_0 = _loop_coords(size, sigma, r=12.5)
    res_1 = _loop_coords(size, sigma, r=12.5)
    res_0 = res_0 + [-12.5, 0]
    res_1 = res_1 + [12.5, 0]
    result = np.concatenate([res_0, res_1])
    return result


def simulate_3_loop(sigma, size):
    pass


def simulate_4_loop(sigma, size):
    pass


def simulate_5_loop(sigma, size):
    pass



# ----------------------------
# Persistent homology analysis
# ----------------------------

def loop_data_frame(sigmas, sizes, save_path, n=30):
    df = {
        'distribution' : [],
        'sigma' : [],
        'size' : [],
        'sample_ID' : [],
        'x' : [], 
        'y' : [], 
    }
    func_dict = {
        'disc' : simulate_disc, 
        '1-loop' : simulate_1_loop,
        '2-loop' : simulate_2_loop
    }
    dists = list(func_dict.keys())
    sample_ID = 0
    its = len(sizes) * len(sigmas) * len(dists) * n
    with tqdm(total=its) as progress:
        for sigma in sigmas:
            for size in sizes:
                for dist in dists:
                    func = func_dict[dist]
                    for i in range(n):
                        result = func(sigma, size)
                        d = [dist, ] * len(result)
                        sig = [sigma, ] * len(result)
                        sz = [size, ] * len(result)
                        df['distribution'] = np.concatenate([df['distribution'], d])
                        df['sigma'] = np.concatenate([df['sigma'], sig])
                        df['size'] = np.concatenate([df['size'], sz])
                        df['x'] = np.concatenate([df['x'], result[:, 0]])
                        df['y'] = np.concatenate([df['y'], result[:, 1]])
                        sid = [sample_ID, ] * len(result)
                        df['sample_ID'] = np.concatenate([df['sample_ID'], sid])
                        sample_ID += 1
                        progress.update(1)
    df = pd.DataFrame(df)
    df.to_csv(save_path)
    return df



def persistent_homology_for_sims(df, save_path, cutoffs=(1, 1, 10), bootstrap=False, subsample=300, n_subsamples=30):
    out = {
        'distribution' : [],
        'sigma' : [],
        'size' : [],
        'sample_ID' : [],
        'mean' : [], 
        'std' : [], 
        'persistence' : [],
        'donutness' : [], 
        'Q1' : [], 
        'Q2' : [], 
        'Q3' : [], 
        'Q4' : [], 
        'second_persistence' : [], 
        'second_donutness' : [], 
    }
    if bootstrap:
        out['subsample_number'] = []
    for i in range(cutoffs[0], cutoffs[1], cutoffs[2]):
        out[f'number_over_{i}'] = []
    its = 0
    for k, grp in df.groupby(['distribution', 'sigma', 'size', 'sample_ID']):
        its += 1
    with tqdm(total=its) as progress:
        for k, grp in df.groupby(['distribution', 'sigma', 'size', 'sample_ID']):
            X = grp[['x', 'y']].values
            if not bootstrap:
                out = get_donut_data(X, k, out, cutoffs)
            else:
                for i in range(n_subsamples):
                    sample_max_idx = k[2] - 1
                    idxs = np.random.randint(0, sample_max_idx, size=subsample)
                    sample = X[idxs, :]
                    out = get_donut_data(sample, k, out, cutoffs, i)
            progress.update(1)
    out = pd.DataFrame(out)
    if save_path is not None:
        out.to_csv(save_path)
    return out



def persistent_homology_for_exp(df, save_path, cutoffs=(1, 1, 10), bootstrap=False, subsample=300, n_subsamples=30, x_col='x_s', y_col='ys'):
    out = {
        'path' : [],
        'frame' : [],
        'mean' : [], 
        'std' : [], 
        'persistence' : [],
        'donutness' : [], 
        'Q1' : [], 
        'Q2' : [], 
        'Q3' : [], 
        'Q4' : [], 
        'second_persistence' : [], 
        'second_donutness' : [], 
    }
    if bootstrap:
        out['subsample_number'] = []
    for i in range(cutoffs[0], cutoffs[1], cutoffs[2]):
        out[f'number_over_{i}'] = []
    its = 0
    for k, grp in df.groupby(['path', 'frame']):
        its += 1
    with tqdm(total=its) as progress:
        for k, grp in df.groupby(['path', 'frame']):
            X = grp[[x_col, y_col]].values
            if not bootstrap:
                out = get_donut_data(X, k, out, cutoffs)
            else:
                for i in range(n_subsamples):
                    sample_max_idx = len(X) - 1
                    idxs = np.random.randint(0, sample_max_idx, size=subsample)
                    sample = X[idxs, :]
                    out = get_donut_data(sample, k, out, cutoffs, i)
            progress.update(1)
    out = pd.DataFrame(out)
    if save_path is not None:
        out.to_csv(save_path)
    return out


def get_donut_data(points, k, out, cutoffs, i=None):
    dgms = ripser(points)['dgms']
    h1 = dgms[1]
    diff = h1[:, 1] - h1[:, 0]
    if len(diff) > 0:
        i = np.argmax(diff)
        max_loop = diff[i]
        mean = np.mean(diff)
        std = np.std(diff)
        donutness = (max_loop - mean) / std
        q1 = scoreatpercentile(diff, 25)
        q2 = scoreatpercentile(diff, 50)
        q3 = scoreatpercentile(diff, 75)
        q4 = scoreatpercentile(diff, 100)
        idxs = np.argsort(diff)
    else:
        max_loop = np.NaN
        mean = np.NaN
        std = np.NaN
        donutness = np.NaN
        q1 = np.NaN
        q2 = np.NaN
        q3 = np.NaN
        q4 = np.NaN
    if len(diff) > 1:
        mns2_persistence = diff[idxs[-2]]
        mns2_donutness = (mns2_persistence - mean) / std
    else:
        mns2_persistence = np.NaN
        mns2_donutness = np.NaN
    keys = list(out.keys())
    if 'distribution' in keys:
        out['distribution'].append(k[0])
        out['sigma'].append(k[1])
        out['size'].append(k[2])
        out['sample_ID'].append(k[3])
    elif 'path' in keys:
        out['path'].append(k[0])
        out['frame'].append(k[1])
    out['mean'].append(mean)
    out['std'].append(std)
    out['donutness'].append(donutness)
    out['persistence'].append(max_loop)
    out['Q1'].append(q1)
    out['Q2'].append(q2)
    out['Q3'].append(q3)
    out['Q4'].append(q4)
    out['second_donutness'].append(mns2_donutness)
    out['second_persistence'].append(mns2_persistence)
    for i in range(cutoffs[0], cutoffs[1], cutoffs[2]):
        upper  = mean + (std * i)
        idxs = np.where(diff > upper)[0]
        n = len(idxs)
        out[f'number_over_{i}'].append(n)
        #outliers = diff[idxs]
    if i is not None:
        out['subsample_number'] = i
    return out



def plot_averages_matrix(ph_data, y_var='donutness', x_var='sigma', hue='distribution', plots='size'):
    vals = pd.unique(ph_data[plots])
    fig, axs = plt.subplots(1, len(vals), sharey=True, sharex=True)
    for i, ax in enumerate(axs.ravel()):
        data = ph_data[ph_data[plots] == vals[i]]
        sns.lineplot(x=x_var, y=y_var, hue=hue, data=data, ax=ax)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        ax.set_title(f'{plots}: {vals[i]}')
    fig.subplots_adjust(right=0.93, left=0.054, bottom=0.102, top=0.942, wspace=0.41)
    plt.show()



def plot_several_y(ph_data, y_name='donutness', y_vars=('donutness', 'second_persistence'), x_var='size', hue='distribution', plots='sigma'):
    vals = pd.unique(ph_data[plots])
    fig, axs = plt.subplots(1, len(vals), sharey=True, sharex=True)
    data_all = {
        x_var : np.concatenate([ph_data[x_var].values for _ in range(len(y_vars))]), 
        y_name : np.concatenate([ph_data[y].values for y in y_vars]), 
        'variables' : np.concatenate([[v,] * len(ph_data) for v in y_vars] ),
        hue:  np.concatenate([ph_data[hue].values for _ in range(len(y_vars))]), 
        plots:  np.concatenate([ph_data[plots].values for _ in range(len(y_vars))]),
    }
    data_all = pd.DataFrame(data_all)
    for i, ax in enumerate(axs.ravel()):
        data = data_all[data_all[plots] == vals[i]]
        sns.lineplot(x=x_var, y=y_name, hue=hue, style='variables', data=data, ax=ax)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        ax.set_title(f'{plots}: {vals[i]}')
    fig.subplots_adjust(right=0.93, left=0.054, bottom=0.102, top=0.942, wspace=0.41)
    plt.show()


def plot_experiment(ph_data, x='frame', y='donutness', y_vars=('donutness', 'second_donutness')):
    data_all = {
        x : np.concatenate([ph_data[x].values for _ in range(len(y_vars))]), 
        y : np.concatenate([ph_data[y].values for y in y_vars]), 
        'variables' : np.concatenate([[v,] * len(ph_data) for v in y_vars] ),
    }
    data_all = pd.DataFrame(data_all)
    sns.lineplot(data=data_all, x=x, y=y, style='variables')
    plt.show()



def scaled_xyz_coordinates(df, rho_col='rho', phi_col='phi', theta_col='theta', x_col='x_s', y_col='ys', ):
    df = _prep_df(df, phi_col)
    df = scale_data(df, rho_col)
    # y = rho * sin(theta) * sin(phi)
    # x = rho * sin(theta) * cos(phi)
    df_px = df[df['x_s'] > 0]
    add_scaled_coords(df, df_px, 'x', '+', rho_col, phi_col, theta_col, x_col)
    df_nx = df[df['x_s'] < 0]
    add_scaled_coords(df, df_nx, 'x', '-', rho_col, phi_col, theta_col, x_col)
    df_py = df[df['ys'] > 0]
    add_scaled_coords(df, df_py, 'y', '+', rho_col, phi_col, theta_col, y_col)
    df_ny = df[df['ys'] < 0]
    add_scaled_coords(df, df_py, 'y', '-', rho_col, phi_col, theta_col, y_col)
    return df



def scale_x_and_y(df, x_col='x_s', y_col='ys'):
    df = df[df['nrtracks'] > 10]
    if 'nb_density_15_pcntf' not in df.columns.values:
        df = quantile_normalise_variables_frame(df, ('nb_density_15', ))
    df = scale_data(df, x_col)
    df = scale_data(df, y_col)
    return df



def _prep_df(df, phi_col='phi'):
    df = df[df['nrtracks'] > 10]
    if 'pid' not in df.columns.values:
        df['pid'] = range(len(df))
    if phi_col not in df.columns.values:
        print('finding phi')
        df = spherical_coordinates(df)
    if 'nb_density_15_pcntf' not in df.columns.values:
        df = quantile_normalise_variables_frame(df, ('nb_density_15', ))
    return df



def add_scaled_coords(full_df, df, coord, sign, rho_col, phi_col, theta_col, coord_col):
    new = get_scaled_coords(df, coord, sign, rho_col, phi_col, theta_col)
    idxs = df.index.values
    full_df.loc[idxs, f'{coord_col}_scaled'] = new



def get_scaled_coords(df, coord, sign, rho_col, phi_col, theta_col):
    # y = rho * sin(theta) * sin(phi)
    # x = rho * sin(theta) * cos(phi)
    if coord == 'y':
        new = df[f'{rho_col}_scaled'].values * np.sin(df[phi_col].values) * np.sin((df[theta_col].values + 0.5 * np.pi))
    elif coord == 'x':
        new = df[f'{rho_col}_scaled'].values * np.cos(df[phi_col].values) * np.sin((df[theta_col].values + 0.5 * np.pi))
    if sign == '-':
        new = - new
    return new
        



def scale_according_to_r(df, rho_col='rho'):
    # anterior
    ant_df = df[df['ys'] > 0]
    # anterior right
    antpx_df = ant_df[ant_df['x_s']>0]
    antpx_df = scale_data(antpx_df)
    # anterior left
    antnx_df = ant_df[ant_df['x_s']<0]
    antnx_df = scale_data(antnx_df)
    # posterior
    pos_df = df[df['ys'] < 0]
    # posterior right
    pospx_df = pos_df[pos_df['x_s'] > 0]
    # posterior right front
    pospxF_df = pos_df[pos_df['phi'] > -0.78539]
    pospxF_df = scale_data(pospxF_df)
    # posterior right back
    pospxB_df = pos_df[pos_df['phi'] < -0.78539]
    pospxB_df = scale_data(pospxB_df)
    # posterior left
    posnx_df = pos_df[pos_df['x_s'] < 0]
    # posterior left front
    posnxF_df = pos_df[pos_df['phi'] > -0.78539]
    posnxF_df = scale_data(posnxF_df)
    # posterior left back
    posnxB_df = pos_df[pos_df['phi'] < -0.78539]
    posnxB_df = quantile_normalise_variables_frame(posnxB_df, ('dist_c', ))
    # concat
    df = pd.concat([antpx_df, antnx_df, pospxF_df, pospxB_df, posnxF_df, posnxB_df])
    df = df.reset_index(drop=True)
    return df



def scale_data(df, col):
    for k, g in df.groupby(['path', 'frame']):
        scaler = StandardScaler()
        data = g[col].values
        data = np.expand_dims(data, 1)
        scaler.fit(data)
        new = scaler.transform(data)
        new = np.squeeze(new)
        idx = g.index.values
        df.loc[idx, f'{col}_scaled'] = new
    return df


def scale_data_1(df, rho_col):
    df = quantile_normalise_variables(df, [rho_col, ])
    new = df[f'{rho_col}_pcnt']
    df[f'{rho_col}_scaled'] = new
    return df


def plot_frames(df, x, y, frames, path):
    df = df[df['path'] == path]
    fig, axs = plt.subplots(1, len(frames), sharey=True, sharex=True)
    for i, ax in enumerate(axs.ravel()):
        f = frames[i]
        data = df[df['frame'] == f]
        sns.scatterplot(x=x, y=y, data=data, ax=ax)
    plt.show()



if __name__ == '__main__':
    sigmas = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)
    sizes = (100, 200, 400, 800, 1600)
    #view_scatter_matrix(simulate_disc, sigmas, sizes) 
    #view_scatter_matrix(simulate_1_loop, sigmas, sizes) 
    #view_scatter_matrix(simulate_2_loop, sigmas, sizes) 
    save_path_0 = '/Users/amcg0011/Data/platelet-analysis/TDA/simulations/simulated_disc_1loop_2loop.csv'
    #df = loop_data_frame(sigmas, sizes, save_path_0, n=30)
    #df = pd.read_csv(save_path_0)
    #sub_df = df[(df['sigma'] > 0.30) & (df['sigma'] < 0.50)] # 3 sigmas
    #sub_df =  sub_df[(sub_df['size'] > 200) & (sub_df['size'] < 1600)] # 2 sizes - big so long time to comute
    save_path = '/Users/amcg0011/Data/platelet-analysis/TDA/simulations/simulated_disc_1loop_2loop_PH-analysis_high-sigma_bootstrapped.csv'
    #out = persistent_homology_for_sims(sub_df, save_path_1, cutoffs=(1, 1, 10))
    #out = persistent_homology_for_sims(df, save_path, cutoffs=(1, 1, 10), bootstrap=True, subsample=300, n_subsamples=30)
    #
    #out = pd.read_csv(save_path)
    #plot_averages_matrix(out, y_var='donutness', x_var='sigma', hue='distribution')
    #plot_averages_matrix(out, y_var='second_donutness', x_var='sigma', hue='distribution')
    #plot_averages_matrix(out, y_var='persistence', x_var='sigma', hue='distribution')
    #plot_averages_matrix(out, y_var='second_persistence', x_var='sigma', hue='distribution')
    #plot_averages_matrix(out, y_var='donutness', x_var='size', hue='distribution', plots='sigma')
    #plot_averages_matrix(out, y_var='second_persistence', x_var='size', hue='distribution', plots='sigma')
    #sigmas = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4)
    #sizes = (100, 200, 400, 800)
    #view_scatter_matrix(simulate_2_loop, sigmas, sizes) 
    #plot_several_y(out, y_name='donutness', y_vars=('donutness', 'second_persistence'), x_var='size', hue='distribution', plots='sigma')
    #plot_several_y(out, y_name='persistence', y_vars=('persistence', 'second_donutness'), x_var='size', hue='distribution', plots='sigma')
    #out['donnutness_1'] = out['donutness'] - out['second_persistence']
    #out['persistence_1'] = out['persistence'] - out['second_donutness']
    #plot_several_y(out, y_name='donutness', y_vars=('donnutness_1', 'donutness'), x_var='size', hue='distribution', plots='sigma')
    #plot_several_y(out, y_name='persistence', y_vars=('persistence_1', 'persistence'), x_var='size', hue='distribution', plots='sigma')
    #plot_several_y(out, y_name='persistence', y_vars=('mean', 'std'), x_var='size', hue='distribution', plots='sigma')

    df_dir = '/Users/amcg0011/Data/platelet-analysis/dataframes'
    saline_n = '211206_saline_df_220827_amp0.parquet'
    saline_p = os.path.join(df_dir, saline_n)
    #saline_exp = '200527_IVMTR73_Inj4_saline_exp3'
    #saline_exp = '191128_IVMTR33_Inj3_saline_exp3'
    #df = pd.read_parquet(saline_p)
    #df = df[df['path'] == saline_exp]
    #df = df[df['nb_density_15_pcntf'] > 50]
    #s_dir = '/Users/amcg0011/Data/platelet-analysis/TDA/simulations'
    #saline_save = '/Users/amcg0011/Data/platelet-analysis/TDA/simulations/single-saline-example-1_dt75_ss200_ns100.csv'
    #out = persistent_homology_for_exp(df, saline_save, cutoffs=(1, 1, 10), bootstrap=True, subsample=200, n_subsamples=100)
    #out['donnutness_1'] = out['donutness'] - out['second_persistence']
    #out['persistence_1'] = out['persistence'] - out['second_donutness']
    #out = pd.read_csv(saline_save)
    #plot_experiment(out, x='frame', y='donutness', y_vars=('donutness', 'second_donutness'))
    #plot_experiment(out, x='frame', y='persistence', y_vars=('persistence', 'second_persistence'))

    # SCALED
    # ------

    #df = scaled_xyz_coordinates(df)
    #df = scale_x_and_y(df, x_col='x_s', y_col='ys')
    #df = df[df['nb_density_15_pcntf'] > 50]
    #saline_save = '/Users/amcg0011/Data/platelet-analysis/TDA/simulations/single-saline-example-1_scaledxy_dt50_ss200_ns100.csv'
    #plot_frames(df, 'x_s_scaled', 'ys_scaled', (10, 30, 50, 100), saline_exp)
    #out = persistent_homology_for_exp(df, saline_save, cutoffs=(1, 1, 10), bootstrap=True, subsample=200, n_subsamples=100, x_col='x_s_scaled', y_col='ys_scaled')
    #out = pd.read_csv(saline_save)

    #plot_experiment(out, x='frame', y='donutness', y_vars=('donutness', 'second_donutness'))
    #out['donutness_1'] = out['donutness'] - out['second_donutness']
    #plot_experiment(out, x='frame', y='donutness', y_vars=('donutness_1', 'donutness'))

    # ALSO CANG - small clots therfore limit sample size
    cang_n = '211206_cang_df.parquet'
    cang_p = os.path.join(df_dir, cang_n)
    #cang_exp = '191113_IVMTR26_Inj4_cang_exp3'
    cang_exp = '191101_IVMTR19_Inj4_cang_exp3'
    df = pd.read_parquet(cang_p)
    paths = pd.unique(df['path'])
    df = df[df['path'] == cang_exp]
    df = scale_x_and_y(df, x_col='x_s', y_col='ys')
    df = df[df['nb_density_15_pcntf'] > 50]
    plot_frames(df, 'x_s_scaled', 'ys_scaled', (10, 30, 50, 100), cang_exp)
    cang_save = '/Users/amcg0011/Data/platelet-analysis/TDA/simulations/single-cang-example-1_scaledxy_dt50_ss200_ns100.csv'
    out = persistent_homology_for_exp(df, cang_save, cutoffs=(1, 1, 10), bootstrap=True, subsample=200, n_subsamples=100, x_col='x_s_scaled', y_col='ys_scaled')
    plot_experiment(out, x='frame', y='donutness', y_vars=('donutness', 'second_donutness'))
    #out = pd.read_csv(cang_save)
    out['donutness_1'] = out['donutness'] - out['second_donutness']
    plot_experiment(out, x='frame', y='donutness', y_vars=('donutness_1', 'donutness'))


    # ALSO BIVA & maybe DMS0

# donutness and persistence should be adjusted acording to the persistence of the second largest loop
# donutness and persistence should be adjusted according to sample size according to the relationship at a represetitive noise level sigma ~ 35% 
# alternatively, points could be subsampled many times with a specified n to produce a bootstrapped estimate for each 
