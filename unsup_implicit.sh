##!/usr/bin/bash

. /home/zqiu/miniconda3/etc/profile.d/conda.sh
conda activate main

# module load cuda/11.3

seeds=(10094 16734 20058 26284 27026)

for x in ${seeds[@]}; do
  echo 'Seed' $x
  python train.py --idx=$1
done

printf "\n"
printf "\n"
