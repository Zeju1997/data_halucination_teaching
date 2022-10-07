##!/usr/bin/bash

# conda activate main

# module load cuda/11.3

seeds=(10432 10756 17913 18147 19003 25821 27732 31154 32367 44112 46852 57595 58431 65731 65800 78957 81860 88415 95873 96693)

for x in ${seeds[@]}; do
  echo 'Seed' $x
  python train.py --seed=$x
done

printf "\n"
printf "\n"
