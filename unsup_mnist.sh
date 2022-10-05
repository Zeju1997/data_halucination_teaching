##!/usr/bin/bash

. /home/zqiu/miniconda3/etc/profile.d/conda.sh
conda activate main

# module load cuda/11.3

# seeds=(65800 27026 95873 45069 99244 86091 51626 84215 10094 48495)
seeds=(16734 26284 48457 70293 10714 75402 87150 20058 57890 97577)


for x in ${seeds[@]}; do
  echo 'Seed' $x
  python train.py --seed=$x
done

printf "\n"
printf "\n"
