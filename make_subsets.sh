#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=MaxMemPerNode
#SBATCH --mem-per-cpu=MaxMemPerCPU
for i in `seq 0.2 0.1 0.9`
do
	python subset.py ./data/combined.csv activity $i ./data/$i"subset".csv
done