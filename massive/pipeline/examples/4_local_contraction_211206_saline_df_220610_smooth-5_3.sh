#!/bin/bash
#SBATCH --job-name=SmoothedLocalContraction
#SBATCH --account=rl54
#SBATCH --time=24:00:00
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --mail-user=Abigail.McGovern1@monash.edu
#SBATCH --mail-type=ALL

source /projects/rl54/Abi/miniconda/bin/activate
conda activate platelets

python /fs02/rl54/Abi/platelet-analysis/massive/pipeline/4_local_contraction.py -f /fs03/rl54/smoothed_dataframes/211206_saline_df_220610_smooth-5_3.parquet -s /fs03/rl54/smoothed_dataframes/211206_saline_df_220610_smooth-5_4.parquet
