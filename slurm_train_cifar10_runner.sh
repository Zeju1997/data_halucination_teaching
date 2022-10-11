#!/bin/bash
# Run sbatch for different seed, model, and experiments

config="cifar10.yaml"
seeds=(10094 20058 27026 48495 65800)
declare -a models=("CNN3" "CNN6" "CNN9" "CNN15")
declare -a experiments=("SGD" "Student" "Baseline")


for experiment in "${experiments[@]}"
do
    for model in "${models[@]}"
    do
        for seed in "${seeds[@]}"
        do
            echo "--------------------------------------------"
            echo "sbatch slurm_train.sh $config $seed $model $experiment"
            echo "--------------------------------------------"
            sbatch slurm_train.sh $config $seed $model $experiment
        done
    done
done