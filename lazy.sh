##!/usr/bin/bash

. /home/zqiu/miniconda3/etc/profile.d/conda.sh
conda activate main

# module load cuda/11.3

# epsilons=(0.0 0.0002 0.0005 0.0008 0.001 0.0015 0.002 0.003 0.01 0.1 0.3 0.5 1.0)
epsilons=(0.002 0.003 0.01 0.1 0.3 0.5 1.0)

for eps in ${epsilons[@]}; do
  echo 'Epsilon' $eps
  python train.py --epsilon=$eps
done


printf "\n"
printf "\n"
