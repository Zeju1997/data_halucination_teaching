##!/usr/bin/bash

# conda activate main

# module load cuda/11.3

seeds=(42049 80553 94094 91291 81639 25730 70525 66170 51420 11468 40075 18066 34478 23457 48294 86239 65800 95873 43852 22442 44597 27026 58431 78957 43886)

for x in ${seeds[@]}; do
  echo 'Seed' $x
  python train.py --seed=$x
done

printf "\n"
printf "\n"
