import numpy as np
import matplotlib.pyplot as plt
import napari
from toolz import curry
from datetime import datetime
import os



def show_platelet(
        df, 
        save_dir,
        vars=[
            ['phi', 'rho', 'theta', 'nb_cont_15'], 
            ['dv', 'zs', 'nb_density_15', 'emb_prox_k5'],
            ['fibrin_dist', 'nb_ca_diff_15', 'ca_corr', 'nb_ca_corr_15'] 
              ],
        sample_var='nrtracks', 
        sample_logic='>', 
        value=180, 
    ):
    # sample platelet
    platelet_df = new_platelet(df, vars, sample_var, sample_logic, value)
    platelet = CurrentPlatelet(platelet_df)
    # plot the first platlet
    fig, axs = plt.subplots(nrows=vars.shape[0], ncols=vars.shape[1])
    fig, axs = plot_vars_over_time(df, vars, platelet, fig, axs)
    # bind sampling to key so that the user can look though as many platelets as are required
    key_press_func = on_key_press(df, sample_var, sample_logic, value, save_dir)
    fig.canvas.mpl_connect('key_press_event', key_press_func)
    # show plot
    plt.show()


def new_platelet(
        df, 
        variable='nrtracks', 
        logic='>', 
        value=180, 
    ):
    cmd = f'df = df[df[{variable}] {logic} {value}]'
    exec(cmd)
    df = df.set_index(['path', 'particle'])
    sample = df.sample(n=1)
    idx = sample.index.values[0]
    platelet_df = df[idx, :].reset_index()
    return platelet_df


def plot_vars_over_time(df, vars, fig, axs):
    vars = np.array(vars)
    for i, ax in enumerate(axs.ravel()):
        var = vars[i]
        ax.plot(df['frame'], df[var])
        ax.set_xlabel('frame')
        ax.set_ylabel(var)
    return fig, axs


@curry
def on_key_press(
        df, 
        vars, 
        platelet,
        variable, 
        logic, 
        value, 
        save_dir, 
        event
    ):
    fig = event.canvas.figure
    ax = event.inaxes or fig.axes[0]
    if event.key == 'up':
        new = new_platelet(df, variable=variable, by=logic, value=value)
        platelet.df = new
        plot_vars_over_time(platelet.df, vars, fig, ax)
    elif event.key == 's':
        now = datetime.now()
        dt = now.strftime("%y%m%d_%H%M%S")
        path = platelet.df.loc[0, 'path']
        particle = platelet.df.loc[0, 'particle']
        n = f'{path}_{particle}_{dt}.csv'
        p = os.path.join(save_dir, n)
        platelet.df.to_csv(p)



class CurrentPlatelet():
    def __init__(self, platelet_df) -> None:
        self._df = platelet_df

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, new):
        self._df = new