##!/usr/bin/bash

# conda activate main

# module load cuda/11.3

seeds=(86239 95873 22442 65800 27026 58431 78957 43886 61489 54162)

for x in ${seeds[@]}; do
  echo 'Seed' $x
  python train.py --seed=$x
done

printf "\n"
printf "\n"
