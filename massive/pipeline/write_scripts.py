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



pipeline_dict = {
    '0_smooth_coords.py' : {
        'name' : 'SmoothPlateletCoords', 
        'time_hrs' : 2, 
        'memory' : 16, 
        'n_tasks' : 1, 
        'env' : 'platelets'
    }, 
    '1_spherical_coords.py' : {
        'name' : 'SmoothedSphericalCoords', 
        'time_hrs' : 3, 
        'memory' : 16, 
        'n_tasks' : 1, 
        'env' : 'platelets'
    },
    '2_dynamic_measures.py' : {
        'name' : 'SmoothedDynamicMeasures', 
        'time_hrs' : 6, 
        'memory' : 16, 
        'n_tasks' : 1, 
        'env' : 'platelets'
    },
    '3_local_environment.py' : {
        'name' : 'SmoothedLocalEnvironment', 
        'time_hrs' : 6, 
        'memory' : 16, 
        'n_tasks' : 1, 
        'env' : 'platelets'
    },
    '4_local_contraction.py' : {
        'name' : 'SmoothedLocalContraction', 
        'time_hrs' : 24, 
        'memory' : 120, 
        'n_tasks' : 12, 
        'env' : 'platelets'
    },
    '5_cumulative.py' : {
        'name' : 'SmoothedCumulative', 
        'time_hrs' : 4, 
        'memory' : 16, 
        'n_tasks' : 1, 
        'env' : 'platelets'
    },
}


def pipeline_script(
        data_path, 
        save_dir,
        save_name, 
        py_path,
        script_dir,
        step_dict

    ):
    s = slurm_sbatch_header(**step_dict)
    save_path = os.path.join(save_dir, save_name)
    s = s + f'python {py_path} -f {data_path} -s {save_path}\n'
    n = Path(py_path).stem + '_' + Path(data_path).stem + '.sh'
    script_path = os.path.join(script_dir, n)
    with open(script_path, 'w') as script:
        script.write(s)
    return save_path


def write_pipeline(
        input_data_path,
        python_dir,
        script_dir,
        scratch_dir, 
        output_dir,
        pipeline_dict=pipeline_dict,
        suffix = 'smooth-5'

    ):
    n = Path(input_data_path).stem
    save_name_base = n + f'_{suffix}'
    data_path = input_data_path
    last_step = len(pipeline_dict) - 1
    counter = 0
    for py_file in pipeline_dict.keys():
        py_path = os.path.join(python_dir, py_file)
        step_dict = pipeline_dict[py_file]
        if counter < last_step:
            save_name = save_name_base + f'_{counter}.parquet'
            data_path = pipeline_script(data_path, scratch_dir, save_name, py_path, script_dir, step_dict)
        else:
            save_name = save_name_base + '.parquet'
            data_path = pipeline_script(data_path, output_dir, save_name, py_path, script_dir, step_dict)
        counter += 1
    return data_path


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('-f', '--file', help='parquet file platelet info')
    args = p.parse_args()

    input_data_path = args.file
    python_dir = '/fs02/rl54/Abi/platelet-analysis/massive/pipeline'
    script_dir = '/fs02/rl54/scripts/smooth_data'
    scratch_dir = '/fs03/rl54/smoothed_dataframes'
    output_dir = '/fs02/rl54/smoothed_dataframes'

    output_path = write_pipeline(input_data_path, python_dir, script_dir, scratch_dir, output_dir)

    # python /fs02/rl54/Abi/platelet-analysis/massive/pipeline/write_scripts.py -f /fs02/rl54/dataframes/211206_biva_df.parquet #
    # python /fs02/rl54/Abi/platelet-analysis/massive/pipeline/write_scripts.py -f /fs02/rl54/dataframes/211206_cang_df.parquet #
    # python /fs02/rl54/Abi/platelet-analysis/massive/pipeline/write_scripts.py -f /fs02/rl54/dataframes/211206_ctrl_df.parquet #
    # python /fs02/rl54/Abi/platelet-analysis/massive/pipeline/write_scripts.py -f /fs02/rl54/dataframes/211206_mips_df.parquet #
    # python /fs02/rl54/Abi/platelet-analysis/massive/pipeline/write_scripts.py -f /fs02/rl54/dataframes/211206_par4--_df.parquet #
    # python /fs02/rl54/Abi/platelet-analysis/massive/pipeline/write_scripts.py -f /fs02/rl54/dataframes/211206_par4--biva_df.parquet #
    # python /fs02/rl54/Abi/platelet-analysis/massive/pipeline/write_scripts.py -f /fs02/rl54/dataframes/211206_sq_df.parquet #
    # python /fs02/rl54/Abi/platelet-analysis/massive/pipeline/write_scripts.py -f /fs02/rl54/dataframes/211206_veh-sq_df.parquet #
    # python /fs02/rl54/Abi/platelet-analysis/massive/pipeline/write_scripts.py -f /fs02/rl54/dataframes/211206_salgav_df.parquet #
    # python /fs02/rl54/Abi/platelet-analysis/massive/pipeline/write_scripts.py -f /fs02/rl54/dataframes/211206_salgav-veh_df.parquet #
    # python /fs02/rl54/Abi/platelet-analysis/massive/pipeline/write_scripts.py -f /fs02/rl54/dataframes/211206_saline_df.parquet #
    # python /fs02/rl54/Abi/platelet-analysis/massive/pipeline/write_scripts.py -f /fs02/rl54/dataframes/211206_veh-mips_df.parquet #