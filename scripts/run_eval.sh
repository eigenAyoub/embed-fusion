#!/bin/bash


# Check for required arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: ./run_eval.sh model_base run_base tag_name"
    exit 1
fi

model_base="$1"
run_base="$2"
tag_name="$3"



for ep in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 26 27 28 30; do

    echo "> Executing eval.py with checkpoint > $2_ep_0${ep}_$3"

    python eval.py "$1" NFCorpus 1 "$2_ep_0${ep}_$3" 0 0 x "$3-$ep"   
    python eval.py "$1" SciFact  1 "$2_ep_0${ep}_$3" 0 0 x "$3-$ep"   
    python eval.py "$1" ArguAna  1 "$2_ep_0${ep}_$3" 0 0 x "$3-$ep"   

done

