#!/usr/bin/bash
for (( counter=10; counter>0; counter-- ))
do
declare -i x=i
python train.py --seed=$x
done
printf "\n"
