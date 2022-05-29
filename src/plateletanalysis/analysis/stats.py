import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats


def compare_groups(df, group_col, groups, variable_cols):
    if len(groups) == 2:
        c0_df = df[df[group_col] == groups[0]]
        c1_df = df[df[group_col] == groups[1]]
        cdf = pd.concat([c0_df])
        for var in variable_cols:
            v0 = c0_df[var].values
            v1 = c1_df[var].values
            res = stats.ttest_ind(v0, v1)
            print(res)
            sns.barplot(x=group_col, y=var, data=df)