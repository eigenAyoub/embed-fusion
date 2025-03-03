#!/bin/bash

for ep in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 26 27 28 30; do

    arg1="768_768_ep_0${ep}_221301"
    arg2="221301-${ep}"

    python eval.py "$arg1" 768 combined SciFact 1 "$arg2" x     

    #python eval.py "$arg1" 154 snowflake-m NFCorpus 1 "$arg2" x 0
    #python eval.py "$arg1" 154 snowflake-m SciFact  1 "$arg2" x 0 
    #python eval.py "$arg1" 154 snowflake-m ArguAna  1 "$arg2" x 0

    echo "Executed eval.py with ep=${ep}"
done

