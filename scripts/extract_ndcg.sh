#!/bin/bash

# Script: extract_ndcg.sh
# Description: Extracts the ndcg_at_10 metric for a given task across multiple epochs.
# Usage: ./extract_ndcg.sh <TaskName> <InsertNameBase>

# Function to display usage instructions
usage() {
    echo "Usage: $0 <TaskName> <InsertNameBase>"
    echo "Example:"
    echo "  $0 SciFact run-ID-ep"
    exit 1
}

# Check if exactly two arguments (TaskName and InsertNameBase) are provided
if [ "$#" -ne 2 ]; then
    echo "Error: Invalid number of arguments."
    usage
fi

# Assign arguments to variables
TASK_NAME="$1"
INSERT_NAME_BASE="$2"

epochs=(01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)

# Define a function to extract the desired metric using jq
extract_metric() {
    local insert_name="$1"
    local task="$2"
    local metric="$3"

    # Construct the full path to the JSON file
    local json_path="results/${insert_name}/no_model_name_available/no_revision_available/${task}.json"

    # Check if the JSON file exists
    if [ ! -f "$json_path" ]; then
        echo "N/A (File not found: $json_path)"
        return
    fi

    # Use jq to extract the desired metric
    local value
    value=$(jq -r ".scores.test[0].${metric} // \"N/A\"" "$json_path" 2>/dev/null)

    # Check if jq succeeded and the value is not null or empty
    if [ $? -ne 0 ] || [ -z "$value" ] || [ "$value" == "null" ]; then
        echo "N/A (Metric not found or invalid)"
    else
        echo "$value"
    fi
}

# Iterate over each epoch and extract the ndcg_at_10 metric
echo "Extracting ndcg_at_10 for task: ${TASK_NAME}"
echo "Insert Name Base: ${INSERT_NAME_BASE}"
echo "--------------------------------------------"

for ep in "${epochs[@]}"; do
    # Construct the insert-name-here argument based on the current epoch
    insert_name="${INSERT_NAME_BASE}-${ep}"

    # Extract the ndcg_at_10 metric
    score=$(extract_metric "$insert_name" "$TASK_NAME" "ndcg_at_10")

    # Display the formatted output
    echo "epoch ${ep}   ndcg_at_10:   ${score} ${insert_name}"
done

