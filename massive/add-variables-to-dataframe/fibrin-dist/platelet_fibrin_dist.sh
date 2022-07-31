#!/bin/bash
#SBATCH --job-name=FibrinSegmentation
#SBATCH --account=rl54
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=Abigail.McGovern1@monash.edu 
#SBATCH --mail-type=ALL

source /projects/rl54/Abi/miniconda/bin/activate
conda activate platelets

python /fs02/rl54/Abi/platelet-analysis/massive/add-variables-to-dataframe/fibrin-dist/platelet_fibrin_dist.py