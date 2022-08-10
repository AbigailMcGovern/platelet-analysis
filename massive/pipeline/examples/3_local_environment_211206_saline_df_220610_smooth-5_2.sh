#!/bin/bash
#SBATCH --job-name=SmoothedLocalEnvironment
#SBATCH --account=rl54
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --mail-user=Abigail.McGovern1@monash.edu
#SBATCH --mail-type=ALL

source /projects/rl54/Abi/miniconda/bin/activate
conda activate platelets

python /fs02/rl54/Abi/platelet-analysis/massive/pipeline/3_local_environment.py -f /fs03/rl54/smoothed_dataframes/211206_saline_df_220610_smooth-5_2.parquet -s /fs03/rl54/smoothed_dataframes/211206_saline_df_220610_smooth-5_3.parquet
