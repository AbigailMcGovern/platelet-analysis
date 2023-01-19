import os
from datetime import datetime
from pathlib import Path
import numpy as np
now = datetime.now()
dt = now.strftime("%y%m%d_%H%M%S")




def slurm_sbatch_header(
        name, 
        time_hrs, 
        memory, 
        n_tasks, 
        n_cpus_per_task=1, 
        email='Abigail.McGovern1@monash.edu',
        gpu=False, 
        partition='m3h', 
        which_gpu='P100:1', 
        env=None, 
        **kwargs
        ):
    # hours
    hrs = np.floor(time_hrs).astype(int)
    if hrs < 10:
        hrs = f'0{hrs}'
    elif 10 < hrs < 100:
        hrs = str(hrs)
    else:
        print(f'{hrs} exceeds maximum... setting hours at maximum (99 hrs). If your job needs more break it down!')
        hrs = '99'
    # minutes and seconds
    min = (time_hrs % 1) * 60
    sec = (min % 1) * 60
    min = np.floor(min).astype(int)
    sec = np.floor(sec).astype(int)
    if min < 10:
        min = f'0{min}'
    else:
        min = str(min)
    if sec < 10:
        sec = f'0{sec}'
    else:
        sec = str(sec)
    
    s = '#!/bin/bash\n'
    s = s + f'#SBATCH --job-name={name}\n'
    s = s + '#SBATCH --account=rl54\n'
    s = s + f'#SBATCH --time={hrs}:{min}:{sec}\n'
    s = s + f'#SBATCH --ntasks={n_tasks}\n'
    s = s + f'#SBATCH --cpus-per-task={n_cpus_per_task}\n'
    s = s + f'#SBATCH --mem={memory}G\n'
    s = s + f'#SBATCH --mail-user={email}\n'
    s = s + '#SBATCH --mail-type=ALL\n\n'
    if gpu:
        s = s + f'#SBATCH --gres=gpu:{which_gpu}\n'
        s = s + f'#SBATCH --partition={partition}\n\n'
    if env is not None:
        s = add_conda_activate(s, env)
    return s



def add_conda_activate(s, env_name):
    s = s + 'source /projects/rl54/Abi/miniconda/bin/activate\n'
    s = s + f'conda activate {env_name}\n\n'
    return s



def write_slurm_for_smooth_patelets(df_dir, script_dir, env='platelets'):
    files = os.listdir(df_dir)
    for f in files:
        path = os.path.join(df_dir, f)
        name = f'SmoothCoords: {f}'
        s = slurm_sbatch_header(name, 12, 16, 1, n_cpus_per_task=1,
                            email='Abigail.McGovern1@monash.edu', 
                            gpu=False, env=env)
        stem = Path(path).stem
        sn = stem + '_smoothed.parquet'
        sp = os.path.join(df_dir, sn)
        s = s + 'python -f {sn} -s {sp}'
        scptn = stem + '_smooth_bash.sh'
        script_path = os.path.join(script_dir, scptn)
        with open(script_path, 'w') as script:
            script.write(s)



if __name__ == '__main__':
    df_dir = '/fs04/rl54/dataframes'
    script_dir = '/fs04/rl54/Abi/scripts/smooth_dfs'
    write_slurm_for_smooth_patelets(df_dir, script_dir)

