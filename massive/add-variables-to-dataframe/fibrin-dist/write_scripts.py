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
        gpu=False, 
        email='Abigail.McGovern1@monash.edu',
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


def write_fibrin_script(py_path, script_dir, data_path, image_dir, meta_dir, save_path):
    s = slurm_sbatch_header('FibrinDistance', 10, 100, 1)
    s = s + f'python {py_path} -d {data_path} -i {image_dir} -m {meta_dir} -s {save_path}\n'
    n = Path(py_path).stem + '_' + Path(data_path).stem 
    script_path = os.path.join(script_dir, n)
    with open(script_path, 'w') as script:
        script.write(s)

if __name__ == '__main__':
    py_path = ''
    script_dir = ''

    # change these
    data_path = ''
    image_dir = ''
    meta_dir = ''
    save_path = ''

    write_fibrin_script(py_path, script_dir, data_path, image_dir, meta_dir, save_path)

