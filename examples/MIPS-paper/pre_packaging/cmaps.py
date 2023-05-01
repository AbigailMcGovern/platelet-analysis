import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns


MIPS_order = ['DMSO (MIPS)', 'MIPS']
cang_order = ['saline','cangrelor']#['Saline','Cangrelor','Bivalirudin']
SQ_order = ['DMSO (SQ)', 'SQ']
pal_MIPS  =dict(zip(MIPS_order, sns.color_palette('Blues')[2::3]))

pal_cang = dict(zip(cang_order, sns.color_palette('Oranges')[2::3]))

pal_SQ = dict(zip(SQ_order, sns.color_palette('Greens')[2::3]))

pal1={**pal_MIPS,**pal_cang,**pal_SQ}