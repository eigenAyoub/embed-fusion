#!/bin/bash

# Check for required arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: ./run_eval.sh model_name tag_name"
    exit 1
fi

model_name="$1"
tag_name="$2"

#for task in NFCorpus SciFact ArguAna QuoraRetrieval NQ; do
for task in NFCorpus SciFact ArguAna; do
    python eval.py x 0 "$model_name" "$task" 0 "$tag_name" x 0
done

