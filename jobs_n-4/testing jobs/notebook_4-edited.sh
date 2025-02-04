#!/bin/bash
#SBATCH --account=gluscevi_339
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4GB
#SBATCH --time=01:00:00
#SBATCH --job-name=notebook4
module purge
module load gcc  # Load necessary modules

eval "$(conda shell.bash hook)"
conda activate 21cmfirstclass
cd /home1/rahimieh/21cmFirstCLASS/
python notebook_4-edited.py
