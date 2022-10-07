##!/usr/bin/bash

. /home/zqiu/miniconda3/etc/profile.d/conda.sh
conda activate main

# module load cuda/11.3

# seeds=(65800 27026 95873 45069 99244 86091 51626 84215 10094 48495)
seed=(10094 16734 20058 26284 27026 48457 48495 51626 57890 65800 70293 84215 86091 87150 95873 97577 99244)

for x in ${seeds[@]}; do
  echo 'Seed' $x
  python train.py --seed=$x
done

printf "\n"
printf "\n"
