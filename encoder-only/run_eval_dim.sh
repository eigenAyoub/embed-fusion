#!/bin/bash

# Script: extract.sh
# Description: Executes eval.py for multiple epochs and tasks using a provided arg1.
# Usage: ./extract.sh <arg1>
# Example: ./extract.sh 1930-10039-xxxx

# Check if exactly one argument (arg1) is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <arg1>"
    echo "Example:"
    echo "  $0 1930-10039-xxxx"
    exit 1
fi

arg1="$1"

epochs=(32 64 128 384 512 768 784 812 1024)

for ep in "${epochs[@]}"; do

    # Define arg2 based on the current epoch
    arg2="1920-1024-trunc-${ep}-2MRL-ep-003"

    # Execute the Python script for different tasks
    python eval.py "$arg1" "$ep" all-4 NFCorpus 1 "$arg2"
    python eval.py "$arg1" "$ep" all-4 SciFact  1 "$arg2"
    python eval.py "$arg1" "$ep" all-4 ArguAna  1 "$arg2"

    # Echo the execution status
    echo "Executed eval.py with ep=${ep}"
done

