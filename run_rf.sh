#!/bin/bash
#SBATCH -N 2
#SBATCH -n 4
#SBATCH --mem=MaxMemPerNode
#SBATCH --mem-per-cpu=MaxMemPerCPU
#SBATCH -t 12:00:00
python ml.py ./data/$1"subset".csv ./scores/single/$1"RF"$2.csv RF $2