#!/bin/bash
#SBATCH --job-name=LocalContractionSaline
#SBATCH --account=rl54
#SBATCH --time=20:00:00
#SBATCH --ntasks=12
#SBATCH --mem=120G
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=Abigail.McGovern1@monash.edu 
#SBATCH --mail-type=ALL

source /projects/rl54/Abi/miniconda/bin/activate
conda activate platelets

python /fs02/rl54/Abi/platelet-analysis/massive/add-variables-to-dataframe/local_contraction/local_contraction.py