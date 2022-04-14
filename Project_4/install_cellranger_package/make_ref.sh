#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=MaxMemPerNode
#SBATCH --mem-per-cpu=MaxMemPerCPU

export PATH=/home/jxiao4/work/BCB503/opt/cellranger-6.1.2:$PATH

cellranger mkref --genome=GRCh38 --fasta=genome.fa --genes=genes.gtf \
                 --genome=Covid --fasta=Sars_cov.fa --genes=Sars_cov.gtf