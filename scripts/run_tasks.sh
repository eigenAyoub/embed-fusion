#!/bin/bash

# Check for required arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: ./run_eval.sh model_name tag_name"
    exit 1
fi

model_name="$1"
tag_name="$2"

for task in NFCorpus SciFact ArguAna SCIDOCS QuoraRetrieval; do
    python eval.py "$1" "$task" 0 x 384 0 x  "$2" 
done

