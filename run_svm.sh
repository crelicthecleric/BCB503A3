#!/bin/bash
#SBATCH -N 2
#SBATCH -n 4
#SBATCH --mem=MaxMemPerNode
#SBATCH --mem-per-cpu=MaxMemPerCPU
#SBATCH -t 12:00:00
python ml.py ./data/$1"subset".csv ./scores/$1"SVM".csv SVM