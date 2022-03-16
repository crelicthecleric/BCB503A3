#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=MaxMemPerNode
#SBATCH --mem-per-cpu=MaxMemPerCPU

for file in 0.2subset.csv;do
	python random_forest.py $file

done 