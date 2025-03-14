#!/bin/bash


# Check for required arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: ./run_eval.sh model_base run_base tag_name"
    exit 1
fi

model_base="$1"
run_base="$2"
tag_name="$3"

#for ep in {1..49}; do
for ep in {1..15}; do
    
    if [[ $ep -lt 10 ]]
    then
        ep="0${ep}"
        echo "> Executing eval.py with checkpoint > $2_ep_0${ep}_$3"
    fi

    python eval.py "$1" NFCorpus 1 "$2_ep_0${ep}_$3" 0 0 x "$3-$ep"   
    python eval.py "$1" SciFact  1 "$2_ep_0${ep}_$3" 0 0 x "$3-$ep"   
    python eval.py "$1" ArguAna  1 "$2_ep_0${ep}_$3" 0 0 x "$3-$ep"   
    #python eval.py "$1" SCIDOCS  1 "$2_ep_0${ep}_$3" 384 0 x "$3-384-$ep"   

    #python eval.py "$1" NFCorpus 1 "$2_ep_0${ep}_$3" 512 0 x "$3-$ep-512"   
    #python eval.py "$1" SciFact  1 "$2_ep_0${ep}_$3" 512 0 x "$3-$ep-512"   
    #python eval.py "$1" ArguAna  1 "$2_ep_0${ep}_$3" 512 0 x "$3-$ep-512"   
    #python eval.py "$1" SCIDOCS  1 "$2_ep_0${ep}_$3" 512 0 x "$3-$ep-512"   

    #python eval.py "$1" NFCorpus 1 "$2_ep_0${ep}_$3" 384 0 x "$3-$ep-384-x"   
    #python eval.py "$1" SciFact  1 "$2_ep_0${ep}_$3" 384 0 x "$3-$ep-384-x"   
    #python eval.py "$1" ArguAna  1 "$2_ep_0${ep}_$3" 384 0 x "$3-$ep-384"   
    #python eval.py "$1" SCIDOCS  1 "$2_ep_0${ep}_$3" 384 0 x "$3-$ep-384"   
done

