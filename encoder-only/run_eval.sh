#!/bin/bash

for ep in 01 02 03 04 05 06 07 08 09 10 12 14 16 18 20 25 28 30; do

    arg1="1920_1024_2_ep_0${ep}"
    arg2="1920-1024-2MRL-ep-${ep}"
    arg4="1920-1024-768-2MRL-ep-${ep}"
    arg3="1920-1024-512-2MRL-ep-${ep}"
    
    python eval.py "$arg1" 1024 all-4 NFCorpus 1 "$arg2"
    python eval.py "$arg1" 1024 all-4 SciFact  1 "$arg2"
    python eval.py "$arg1" 1024 all-4 ArguAna  1 "$arg2"
    
    python eval.py "$arg1" 768 all-4 NFCorpus 1 "$arg4"
    python eval.py "$arg1" 768 all-4 SciFact  1 "$arg4"
    python eval.py "$arg1" 768 all-4 ArguAna  1 "$arg4"

    python eval.py "$arg1" 512 all-4 NFCorpus 1 "$arg3"
    python eval.py "$arg1" 512 all-4 SciFact  1 "$arg3"
    python eval.py "$arg1" 512 all-4 ArguAna  1 "$arg3"

    echo "Executed eval.py with ep=${ep}"
done

