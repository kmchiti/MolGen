#!/bin/bash

# Initialize a counter
counter=0
max_submissions=7

while true; do
    # Check if counter has reached the limit
    if [[ "$counter" -ge $max_submissions ]]; then
        echo "Maximum number of submissions ($max_submissions) reached. No more jobs will be submitted."
        break  # Exit the loop
    fi

    # Get all jobs for the user and their statuses
    job_statuses=$(squeue -u kamran.chitsaz | awk 'NR>1 {print $5}')  # Skip the header line and get only the status column

    # Count the number of jobs that are not running
    non_running_jobs=$(echo "$job_statuses" | grep -vc 'R')

    # Check if there are no non-running jobs and at least one job exists
    if [[ "$non_running_jobs" -eq 0 ]] && [[ ! -z "$job_statuses" ]]; then
        # All jobs are running, submit a new job
        sbatch scripts/eval_docking.sh
        echo "All jobs are running. New job submitted at $(date)."
        ((counter++))  # Increment the counter
    else
        echo "Not all jobs are running or no jobs are currently queued at $(date). No new job submitted."
    fi

    # Wait for 60 seconds before checking again
    sleep 60
done
