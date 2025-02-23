#!/bin/bash

latest_folder=$(ls -td results/* | head -1)
json_file=$(find "$latest_folder" -name "NFCorpus.json")
ndcg_score=$(grep -oP '"ndcg_at_10"\s*:\s*\K[0-9.]+' "$json_file")

echo "NDCG@10 score: $ndcg_score"

