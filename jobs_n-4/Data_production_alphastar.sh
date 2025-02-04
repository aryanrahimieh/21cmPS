#!/bin/bash
#SBATCH --account=gluscevi_339
#SBATCH --partition=epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100GB
#SBATCH --time=23:59:59
#SBATCH --job-name=alphastar_Data_production
module purge
module load gcc  # Load necessary modules

eval "$(conda shell.bash hook)"
conda activate 21cmfirstclass
cd /home1/rahimieh/21cmFirstCLASS/idm_runs_with_man_save/
python Data_production_alphastar.py
