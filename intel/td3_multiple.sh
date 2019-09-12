for i in {1..$1}
do
    qsub -n "td3_job_$i" -l walltime=168:00:00 -q extended intel_td3.sh -F "$i"
done


