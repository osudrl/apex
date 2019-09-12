for i in {1..$1}
do
    qsub -n "ppo_job_$i" -l walltime=168:00:00 -q extended intel_ppo.sh -F "$i"
done


