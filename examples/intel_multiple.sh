for i in {1..5}
do
    qsub -l walltime=168:00:00 -q extended intel_single.sh -F "$i"
done