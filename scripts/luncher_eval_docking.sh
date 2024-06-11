#!/bin/bash

# Initialize a counter
counter=0
max_submissions=7

# Submit the first job
sbatch scripts/eval_docking.sh
echo "First job submitted at $(date)."
((counter++))  # Increment the counter

while true; do
    # Check if counter has reached the limit
    if [[ "$counter" -ge $max_submissions ]]; then
        echo "Maximum number of submissions ($max_submissions) reached. No more jobs will be submitted."
        break  # Exit the loop
    fi

    # Wait until all jobs are running
    while true; do
        # Get all jobs for the user and their statuses
        job_statuses=$(squeue -u kamran.chitsaz | awk 'NR>1 {print $5}')  # Skip the header line and get only the status column

        # Count the number of jobs that are not running
        non_running_jobs=$(echo "$job_statuses" | grep -vc 'R')

        # Check if there are no non-running jobs and at least one job exists
        if [[ "$non_running_jobs" -eq 0 ]] && [[ ! -z "$job_statuses" ]]; then
            echo "All jobs are running at $(date)."
            break
        else
            echo "Not all jobs are running or no jobs are currently queued at $(date). Checking again in 10 seconds."
            sleep 10
        fi
    done

    # Wait for 90 seconds before submitting a new job
    echo "Waiting for 90 seconds before submitting a new job."
    sleep 90

    # Submit a new job
    sbatch scripts/eval_docking.sh
    echo "New job submitted at $(date)."
    ((counter++))  # Increment the counter
done
