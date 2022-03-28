#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=MaxMemPerNode
#SBATCH --mem-per-cpu=MaxMemPerCPU
/home/ejraven/anaconda3/bin/python3 basic_stats.py