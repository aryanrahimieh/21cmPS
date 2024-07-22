#!/bin/bash
#SBATCH --account=gluscevi_339
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=4GB
#SBATCH --time=03:00:00
#SBATCH --job-name=21cm1stclass
module purge
module load gcc  # Load necessary modules

eval "$(conda shell.bash hook)"
conda activate 21cmfirstclass
cd /home1/rahimieh/21cmFirstCLASS/
python notebook_3-edited.py
