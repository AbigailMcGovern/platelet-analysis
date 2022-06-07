#!/bin/bash
#SBATCH --job-name=ClusterSCnV
#SBATCH --account=rl54
#SBATCH --time=03:00:00
#SBATCH --ntasks=8
#SBATCH --mem=50G
#SBATCH --cpus-per-task=1

source /projects/rl54/Abi/miniconda/bin/activate
conda activate platelets

python /fs02/rl54/Abi/platelet-analysis/massive/M3_cluster_platelets_1.py