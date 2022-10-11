#!/bin/bash
# Run sbatch for different seed, model, and experiments

config="mnist_blackbox_implicit.yaml"
seeds=(10094 20058 27026 48495 65800)
model="MLP"
declare -a experiments=("SGD" "Student" "Baseline")


for experiment in "${experiments[@]}"
do
    for seed in "${seeds[@]}"
    do
        echo "--------------------------------------------"
        echo "sbatch slurm_train.sh $config $seed $model $experiment"
        echo "--------------------------------------------"
        sbatch slurm_train.sh $config $seed $model $experiment
    done
done