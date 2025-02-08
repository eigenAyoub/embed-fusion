#!/bin/bash

#for ep in 01 02 03 04 05 06 07 08 09 10 12 14 16 18 20 25 28 30; do
for ep in 01 03 06 08 09 10 11 12; do

    arg1="768_154_6_ep_0${ep}"
    arg2="768-154-6MRL-ep-${ep}"
    #arg3="768_512_154_1_ep_0${ep}"
    #arg4="768-512-154-1MRL-ep-${ep}"

    #python eval.py "$arg1" 768 snowflake-m NFCorpus 1 "$arg2"
    #python eval.py "$arg1" 768 snowflake-m SciFact  1 "$arg2"
    #python eval.py "$arg1" 768 snowflake-m ArguAna  1 "$arg2"
    
    python eval.py "$arg1" 154 snowflake-m NFCorpus 1 "$arg2"
    python eval.py "$arg1" 154 snowflake-m SciFact  1 "$arg2"
    python eval.py "$arg1" 154 snowflake-m ArguAna  1 "$arg2"

    #python eval.py "$arg1" 512 all-4 NFCorpus 1 "$arg3"
    #python eval.py "$arg1" 512 all-4 SciFact  1 "$arg3"
    #python eval.py "$arg1" 512 all-4 ArguAna  1 "$arg3"

    echo "Executed eval.py with ep=${ep}"
done

for ep in "NFCorpus" "SciFact" "ArguAna"; do 
    python eval.py x 768 snowflake-m "$ep" " 0 xxyy 
    echo "done with $ep"; 
done

