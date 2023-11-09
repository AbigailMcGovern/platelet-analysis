import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ptitprince as pt


def violin_for_validation(
        data, 
        vars=('VI: GT | Output', 'VI: Output | GT', 'Count difference')
        ):
    ort = "h"; sigma = .4; move = .3
    n_vars = len(vars)
    fig, axs = plt.subplots(1, len(vars), sharey=True)
    order = ('small thrombus', 'medium thrombus', 'large thrombus')
    for ax, v in zip(axs, vars):
        pt.RainCloud(x='thrombus', y=v, hue='thrombus', order=order,
                         hue_order=order, data=data, bw = sigma,
                        width_viol = .7, orient = ort, move=move, alpha=0.8, ax=ax)
        ax.legend([],[], frameon=False)
        sns.despine()
    fig.subplots_adjust(right=0.98, left=0.13, bottom=0.15, top=0.98, wspace=0.165, hspace=0.5)
    fig.set_size_inches(n_vars * 3.5, 3.5)
    plt.show()

def var_names(s):
    if s.find('saline') != -1:
        return 'medium thrombus'
    elif s.find('cang') != -1:
        return 'small thrombus'
    elif s.find('DMSO2') != -1:
        return 'large thrombus'
    

sp = '/Users/abigailmcgovern/Data/platelet-analysis/DL-validation/2306/230630_DLvalidation_scores.csv'
scores = pd.read_csv(sp)
scores['thrombus'] = scores['file'].apply(var_names)
scores['Count difference (%)'] = scores['Count difference (%)'] * 100
violin_for_validation(scores)
