#!/bin/bash
# Script: extract_ndch_tasks.sh
# Description: Extracts the ndcg_at_10 metric for specific tasks from a given model folder.
# Usage: ./extract_ndch_tasks.sh <model_folder>

MODEL_FOLDER="$1"
TASKS=("NFCorpus" "SciFact" "ArguAna" "QuoraRetrieval")
JSON_PATH="results/${MODEL_FOLDER}/no_model_name_available/no_revision_available/"


for TASK in "${TASKS[@]}"; do
    F_PATH="${JSON_PATH}${TASK}.json"
    VALUE=$(jq -r ".scores.test[0].ndcg_at_10 // \"N/A\"" "$F_PATH" 2>/dev/null)
    echo "Task: ${TASK}: ${VALUE}"
done
