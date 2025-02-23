#!/bin/bash

#for ep in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20; do
for ep in 01 02; do

    arg1="f768_256_3_ep_0${ep}"
    arg2="rand_ep_0${ep}"
    
    python eval.py "$arg1" 154 snowflake-m NFCorpus 1 "$arg2" x 0
    #python eval.py "$arg1" 154 snowflake-m SciFact  1 "$arg2" x 0 
    #python eval.py "$arg1" 154 snowflake-m ArguAna  1 "$arg2" x 0

    echo "Executed eval.py with ep=${ep}"
done

