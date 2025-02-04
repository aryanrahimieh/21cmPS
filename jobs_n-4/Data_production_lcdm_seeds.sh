#!/bin/bash
#SBATCH --account=gluscevi_339
#SBATCH --partition=epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50GB
#SBATCH --time=20:00:00
#SBATCH --job-name=seeds_lcdm_Data_production
module purge
module load gcc  # Load necessary modules

eval "$(conda shell.bash hook)"
conda activate 21cmfirstclass
cd /home1/rahimieh/21cmFirstCLASS/idm_runs_with_man_save/
python Data_production_lcdm_seeds.py
