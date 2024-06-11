#!/bin/bash

# List of targets to run experiments on
targets=('parp1' '5ht1b' 'jak2' 'braf')

# Specify the batch size
batch_size=1024

# Initialize a counter
start_index=0

target='fa7'
echo "Submitting job for target $target with batch size $batch_size and start index $start_index"
sbatch scripts/eval_docking.sh $target $batch_size $start_index
echo "Job for target $target submitted at $(date)."

# Loop through each target and submit a job with the given batch size
for target in "${targets[@]}"; do

    # Wait until all jobs are running
    while true; do
        job_statuses=$(squeue -u kamran.chitsaz | awk 'NR>1 {print $5}')
        non_running_jobs=$(echo "$job_statuses" | grep -vc 'R')

        if [[ "$non_running_jobs" -eq 0 ]] && [[ ! -z "$job_statuses" ]]; then
            echo "All jobs are running at $(date) for target $target."
            break
        else
            echo "Not all jobs are running or no jobs are currently queued at $(date) for target $target. Checking again in 10 seconds."
            sleep 10
        fi
    done

    echo "Waiting for 30 seconds before submitting a new job for target $target."
    sleep 30


    echo "Submitting job for target $target with batch size $batch_size and start index $start_index"
    sbatch scripts/eval_docking.sh $target $batch_size $start_index
    echo "Job for target $target submitted at $(date)."

done

echo "All jobs submitted successfully."
