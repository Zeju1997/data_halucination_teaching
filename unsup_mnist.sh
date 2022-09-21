##!/usr/bin/bash

# conda activate main

# module load cuda/11.3

seeds=(65800 27026 95873 45069 99244 86091 51626 84215 10094 48495)

for x in ${seeds[@]}; do
  echo 'Seed' $x
  python train.py --seed=$x
done

printf "\n"
printf "\n"
