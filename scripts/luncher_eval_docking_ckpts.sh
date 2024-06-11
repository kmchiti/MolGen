#!/bin/bash

# List of targets to run experiments on
targets=('fa7' 'parp1' '5ht1b' 'jak2' 'braf')

# Specify the batch size
batch_size=1024

# Initialize a counter
counter=0

# Loop through each target and submit a job with the given batch size
for target in "${targets[@]}"; do
    # Calculate start index
    start_index=$((counter * batch_size))

    echo "Submitting job for target $target with batch size $batch_size and start index $start_index"
    sbatch scripts/eval_docking.sh $target $batch_size $start_index
    echo "Job for target $target submitted at $(date)."

    # Increment the counter
    ((counter++))
done

echo "All jobs submitted successfully."
