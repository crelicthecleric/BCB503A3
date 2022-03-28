for frac in `seq 0.1 0.1 0.9`
do
	sbatch run_sub_zeros.sh $frac
done