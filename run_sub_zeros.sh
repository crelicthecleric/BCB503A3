#!/bin/bash
#SBATCH -N 2
#SBATCH -n 4
#SBATCH --mem=MaxMemPerNode
#SBATCH --mem-per-cpu=MaxMemPerCPU
#SBATCH -t 12:00:00

python substitute_zeros.py ./data/0.5subset.csv ./zeros/$1"zeros".py $1