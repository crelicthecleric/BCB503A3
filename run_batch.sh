for frac in `seq 0.7 0.1 1.0`
do
	sbatch run_svm.sh $frac
	for treeqty in 25 50 100 200 400
	do
		sbatch run_rf.sh $frac $treeqty
	done
done