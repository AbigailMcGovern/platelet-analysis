from plateletanalysis.analysis.peaks_analysis import experiment_data_df, growth_data, find_curves_exp
from plateletanalysis.topology.donutness import summary_donutness
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plateletanalysis.variables.basic import time_seconds, time_tracked_var


p_saline = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/211206_saline_df_spherical-coords.parquet'
saline = pd.read_parquet(p_saline)
saline = time_seconds(saline)
saline = time_tracked_var(saline)
p_saline_donut = '/Users/abigailmcgovern/Data/platelet-analysis/toroidal_paper/data/saline_donut_data_scaled_sn200_n100_c50.csv'
saline_ddf = pd.read_csv(p_saline_donut)
saline_ddf = time_seconds(saline_ddf)
save_path_g = '/Users/abigailmcgovern/Data/platelet-analysis/toroidal_paper/data/231009_saline_growth_data.csv'
saline_gdf = growth_data(saline)

ddf_sum = summary_donutness(saline_ddf)
#find_curves_exp(ddf_sum, 'donutness', w=25, height=1.5, dist=300, t_col='time (s)', prom=0.00001)
#find_curves_exp(saline_gdf, 'net loss (%)', w=None, height=5, t_col='time (s)')
find_curves_exp(saline_gdf, 'platelet count', w=None, height=500, dist=60, t_col='time (s)')


#saline_gdf.to_csv(save_path_g)
save_path_s = '/Users/abigailmcgovern/Data/platelet-analysis/toroidal_paper/data/231009_saline_summary_data.csv'
# res = experiment_data_df(saline, saline_gdf, saline_ddf, save_path_s, True)

#psel_sum = pd.read_csv('/Users/abigailmcgovern/Data/platelet-analysis/P-selectin')