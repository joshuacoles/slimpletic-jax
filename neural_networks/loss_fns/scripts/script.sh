#!/bin/bash

# Set the directory containing the JSON files
dir="$1"
method="$2"

if [ -z "$dir" ]; then
    echo "Usage: $0 <dir> <method>"
    exit 1
fi

if [ -z "$method" ]; then
    echo "Usage: $0 <dir> <method>"
    exit 1
fi

# check method is either loss or iter
if [ "$method" != "loss" ] && [ "$method" != "iter" ]; then
    echo "Method must be either 'loss' or 'iter'"
    exit 1
fi

if [ "$method" = "loss" ]; then
    echo "Checking for convergence based on loss"
    # 3rd arg or default to 0.05
    tolerance=${3:-0.05}
else
    echo "Checking for convergence based on iteration count"
fi

# Initialize counters
converged=0
failed_to_converge=0

# Loop through all JSON files in the directory
for file in "$dir"/*.json; do
    # Check if the file exists
    if [ -f "$file" ]; then
      if [ "$method" == "loss" ]; then
        # We consider something converged if the **difference in loss** between the found and true embeddings is less than 0.2
        count=$( jq ".loss - .true_loss < ${tolerance} and .loss - .true_loss > -${tolerance}" "$file" | grep -c true)
      else
        # We consider something converged if exited at an iteration count less than maxiter.
        count=$(jq '.opt_state.iter_num < .maxiter' "$file" | grep -c true)
      fi

      # We will only get a true / false so this works
      converged=$((converged + count))
      failed_to_converge=$((failed_to_converge + (1 - count)))
    fi
done

# Calculate total cases
total=$((converged + failed_to_converge))

# Calculate percentages
if [ "$total" -gt 0 ]; then
    converged_percentage=$(printf "%.2f" "$(echo "scale=2; 100 * $converged / $total" | bc)")
    failed_to_converge_percentage=$(printf "%.2f" "$(echo "scale=2; 100 * $failed_to_converge / $total" | bc)")

    # Print the results
    echo "Converged: $converged/$total = $converged_percentage%"
    echo "Failed to converge: $failed_to_converge/$total = $failed_to_converge_percentage%"
else
  echo "No data found"
fi
