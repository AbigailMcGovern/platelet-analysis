#!/bin/bash
#SBATCH --job-name=SmoothPlateletCoords
#SBATCH --account=rl54
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --mail-user=Abigail.McGovern1@monash.edu
#SBATCH --mail-type=ALL

source /projects/rl54/Abi/miniconda/bin/activate
conda activate platelets

python /fs02/rl54/Abi/platelet-analysis/massive/pipeline/0_smooth_coords.py -f /fs02/rl54/dataframes/211206_saline_df_220610.parquet -s /fs03/rl54/smoothed_dataframes/211206_saline_df_220610_smooth-5_0.parquet
