from plateletanalysis import local_density, add_neighbour_lists
import pandas as pd

path = "path/to/dataframe.parquet"
df = pd.read_parquet(path)

df = add_neighbour_lists(df)
df = local_density(df)