from plateletanalysis.topology.largest_loop import largest_loop_comparison_data, plot_donut_comparison
import pandas as pd
import numpy as np
import os



sd = '/Users/amcg0011/Data/platelet-analysis/TDA/treatment_comparison'
save_path = os.path.join(sd, '221025_longest-loop-analysis.csv')
data = pd.read_csv(save_path)
#print(pd.unique(data['treatment']))
# ['saline' 'bivalirudin' 'cangrelor' 'SQ' 'MIPS' 'control' 'PAR4--'
 # 'PAR4-- bivalirudin' 'salgav' 'DMSO (salgav)' 'DMSO (SQ)']

# ----------------------
# Make comparitive plots
# ----------------------

plot_1 = ('saline', 'cangrelor', 'bivalirudin')
plot_donut_comparison(data, plot_1)

#plot_2 = ('DMSO (MIPS)', 'MIPS')
#plot_donut_comparison(data, plot_2)

plot_3 = ('SQ', 'DMSO (SQ)')
plot_donut_comparison(data, plot_3)

plot_4 = ('saline', 'bivalirudin', 'PAR4--', 'PAR4-- bivalirudin')
plot_donut_comparison(data, plot_4)

plot_5 = ('salgav', 'DMSO (salgav)')
plot_donut_comparison(data, plot_5)

plot_6 = ('saline', 'control', 'DMSO (MIPS)', 'DMSO (SQ)', 'DMSO (salgav)')
plot_donut_comparison(data, plot_6)