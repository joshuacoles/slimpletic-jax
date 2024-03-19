#!/bin/bash

# Set the directory containing the JSON files
dir="$1"

# Initialize counters
converged=0
failed_to_converge=0

# Loop through all JSON files in the directory
for file in "$dir"/*.json; do
    # Check if the file exists
    if [ -f "$file" ]; then
        # We consider something converged if exited at an iteration count less than maxiter.
        count=$(jq '.opt_state.iter_num < .maxiter' "$file" | grep -c true)
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
