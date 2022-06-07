import os
from datetime import datetime

now = datetime.now()
dt = now.strftime("%y%m%d")


d = '/fs02/rl54/dataframes'
file_names = [
    '211206_cang_df_spherical-coords.parquet', 
    '211206_biva_df_spherical-coords.parquet'
]
py_d = '/fs02/rl54/Abi/platelet-analysis/massive'
py_path = os.path.join(py_d, 'add_vars.py')

for i, f in enumerate(file_names):

    path = os.path.join(d, f)
    n = dt + '_' + f
    save = os.path.join(d, n)

    s = '#!/bin/bash\n'
    s = s + f'#SBATCH --job-name=LocalVars-{f}\n'
    s = s + '#SBATCH --account=rl54\n'
    s = s + '#SBATCH --time=06:00:00\n'
    s = s + '#SBATCH --ntasks=1\n'
    s = s + '#SBATCH --mem=16G\n'
    s = s + '#SBATCH --cpus-per-task=1\n'
    s = s + '#SBATCH --mail-user=Abigail.McGovern1@monash.edu\n'
    s = s + '#SBATCH --mail-type=ALL\n\n'
    s = s + 'source /projects/rl54/Abi/miniconda/bin/activate\n'
    s = s + 'conda activate platelets\n'
    s = s + f'python {py_path} -f {path} -s {save}\n'

    out = os.path.join(py_d, f'{dt}_add_vars_{i}.sh')

    with open(out, 'w') as script:
        script.write(s)