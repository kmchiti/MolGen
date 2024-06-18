#!/bin/bash
#SBATCH --job-name=launcher
#SBATCH --time=0-09:00

#SBATCH --partition=unkillable-cpu    # ask for unkillable job
#SBATCH --cpus-per-task=1              # number of cpus per gpu

# Initialize a counter
counter=0
max_submissions=23

# Specify the batch size
batch_size=4096

# List of targets to iterate over
targets=('fa7' 'parp1' '5ht1b' 'jak2' 'braf')

# Loop through each target and submit a job
for target in "${targets[@]}"; do
    # Calculate start index
    start_index=$((counter * batch_size))
    echo "Submitting job for target $target with batch size $batch_size and start index $start_index"
    sbatch scripts/eval_docking.sh $target $batch_size $start_index
    echo "Job for target $target submitted at $(date)."
    ((counter++))  # Increment the counter

    while true; do
        # Check if counter has reached the limit
        if [[ "$counter" -ge $max_submissions ]]; then
            echo "Maximum number of submissions ($max_submissions) reached. No more jobs will be submitted for target $target."
            break  # Exit the loop for the current target
        fi

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

        # Calculate start index
        start_index=$((counter * batch_size))
        echo "Submitting job for target $target with batch size $batch_size and start index $start_index"
        sbatch scripts/eval_docking.sh $target $batch_size $start_index
        echo "Job for target $target submitted at $(date)."

        ((counter++))
    done

    # Reset counter for next target
    counter=0
done
