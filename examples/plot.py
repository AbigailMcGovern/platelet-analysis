import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from .measure import length_entropy



def volume_distribution(df, vol_col='volume', bins=150):
    df[vol_col].hist(bins=bins)
    plt.show()



def plot_entropy():
    pass