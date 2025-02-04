#!/bin/bash
#SBATCH --account=gluscevi_339
#SBATCH --partition=epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=10GB
#SBATCH --time=05:00:00
#SBATCH --job-name=lcdm_Data_production_sigma
module purge
module load gcc  # Load necessary modules

eval "$(conda shell.bash hook)"
conda activate 21cmfirstclass
cd /home1/rahimieh/21cmFirstCLASS/idm_runs_with_man_save/
python Data_production_lcdm.py
