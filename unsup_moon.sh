##!/usr/bin/bash

# conda activate main

# module load cuda/11.3

seeds=(87017 94827 17913 46852 25759 81737 90830 87999 32367 10756)

for x in ${seeds[@]}; do
  echo 'Seed' $x
  python train.py --seed=$x
done

printf "\n"
printf "\n"
