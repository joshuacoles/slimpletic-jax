#!/usr/bin/env bash

# Set the script to exit on any errors, and pipe failures
set -e -o pipefail

dir="$1"
output_dir="$2"

if [ -z "$dir" ]; then
    echo "Usage: $0 <dir>"
    exit 1
fi
mkdir -p "$output_dir"

for file in "$dir"/*.json; do
    echo "Rendering $file"
    # Replace json extension with png
    /Users/joshuacoles/Developer/checkouts/fyp/slimplectic-jax/loss_fn/render.py "$file" "$output_dir/$(basename "$file" .json).png"
done
