#!/bin/bash

# Script: extract.sh
# Description: Extracts the ndcg_at_10 metric for a given task across multiple specified values.
# Usage: ./extract.sh <TaskName> <InsertNameBase>
# Example: ./extract.sh SciFact 1920-1024

# ---------------------------
# Function Definitions
# ---------------------------

# Function to display usage instructions
usage() {
    echo "Usage: $0 <TaskName> <InsertNameBase>"
    echo "Example:"
    echo "  $0 SciFact 1920-1024"
    echo "  $0 TaskXX 1920-1024"
    exit 1
}

# Function to extract the desired metric using jq
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

# ---------------------------
# Argument Validation
# ---------------------------

# Check if exactly two arguments (TaskName and InsertNameBase) are provided
if [ "$#" -ne 2 ]; then
    echo "Error: Invalid number of arguments."
    usage
fi

# Assign arguments to variables
TASK_NAME="$1"
INSERT_NAME_BASE="$2"

# ---------------------------
# Define Values to Iterate Over
# ---------------------------

# Define the list of values to iterate over (replacing epochs)
values=(32 64 128 384 512 768 784 812 1024)

# ---------------------------
# Main Execution
# ---------------------------

echo "Extracting ndcg_at_10 for task: ${TASK_NAME}"
echo "Insert Name Base: ${INSERT_NAME_BASE}"
echo "--------------------------------------------"

for val in "${values[@]}"; do
    # Construct the insert_name based on the current value
    # Desired format: 1920-1024-trunc-32-8MRL-ep-010
    #insert_name="${INSERT_NAME_BASE}-trunc-${val}-8MRL-ep-010"
    #insert_name="${INSERT_NAME_BASE}-trunc-${val}-4MRL-ep-003"
    insert_name="${INSERT_NAME_BASE}-trunc-${val}-2MRL-ep-003"

    # Extract the ndcg_at_10 metric
    score=$(extract_metric "$insert_name" "$TASK_NAME" "ndcg_at_10")

    # Display the formatted output
    echo "value ${val}   ndcg_at_10:   ${score}"
done

