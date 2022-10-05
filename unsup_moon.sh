##!/usr/bin/bash

# conda activate main

# module load cuda/11.3

seeds=(22442 65800 27026 58431 78957 43886 61489 54162 16734 26284 48457 70293 10714 75402 87150 20058 57890 97577)

for x in ${seeds[@]}; do
  echo 'Seed' $x
  python train.py --seed=$x
done

printf "\n"
printf "\n"
