from plateletanalysis.analysis.clustering import PCA_objects
from plateletanalysis.analysis.scatter import basic_scatter

import pandas as pd
import os


d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
sp = os.path.join(d, '211206_saline_df_SCnV-umap-dbscan_ALL_cleaned.parquet')
df = pd.read_parquet(sp)

df, pca = PCA_objects(df, 'SCnV')
#df.to_parquet(sp)
print(pca.explained_variance_ratio_)
print(pca.noise_variance_)

#sdf = df.sample(frac=0.05)
#pcs_cols = ['PCA_0_SCnV', 'PCA_1_SCnV']
#basic_scatter(sdf, pcs_cols[0], pcs_cols[1], 'phi')