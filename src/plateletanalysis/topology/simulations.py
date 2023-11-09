import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict



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



def simulate_1_loop(sigma, size, r=37.5):
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
    size_0 = np.floor(size / 2).astype(int)
    size_1 = np.ceil(size / 2).astype(int)
    full = size_0 + size_1
    assert size == full
    res_0 = _loop_coords(size_0, sigma, r=12.5)
    res_1 = _loop_coords(size_1, sigma, r=12.5)
    res_0 = res_0 + [-12.5, 0]
    res_1 = res_1 + [12.5, 0]
    result = np.concatenate([res_0, res_1])
    return result



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


def simulate_matching_data(
        data, 
        save_path,
        sample_col='path', 
        time_col='frame', 
        disc_sigma=0.25, 
        loop1_sigma=0.25, 
        loop2_sigma=0.25):
    sim_data = defaultdict(list)
    func_dict = {
        'disc' : (simulate_disc, disc_sigma), 
        '1-loop' : (simulate_1_loop, loop1_sigma),
        '2-loop' : (simulate_2_loop, loop2_sigma)
    }
    for k, grp in data.groupby([sample_col, time_col]):
        n = len(grp)
        sim_data[sample_col] = sim_data[sample_col] + [k[0], ] * n
        sim_data[time_col] = sim_data[time_col] + [k[1], ] * n
        for key in func_dict.keys():
            func = func_dict[key][0]
            sigma = func_dict[key][1]
            coords = func(sigma, n)
            x = list(coords[:, 0])
            x_name = 'x_' + key
            sim_data[x_name] = sim_data[x_name] + x
            y = list(coords[:, 1])
            y_name = 'y_' + key
            sim_data[y_name] = sim_data[y_name] + y
    sim_data = pd.DataFrame(sim_data)
    data.to_csv(save_path)
    return sim_data



        

        

