##!/usr/bin/bash

# conda activate main

# module load cuda/11.3

seeds=(86239 65800 10988 95873 43852 22442 44992 44597 27026 58431)

for x in ${seeds[@]}; do
  echo 'Seed' $x
  python train.py --seed=$x
done

printf "\n"
printf "\n"
