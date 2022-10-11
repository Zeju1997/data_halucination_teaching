#!/bin/bash
#SBATCH --ntasks=1                          # Number of tasks (see below)
#SBATCH --nodes=1                           # Ensure that all cores are on one machine
#SBATCH --partition=cpu-short              # Partition to submit to
#SBATCH --exclude=slurm-bm-[46,49]          # Explicitly exclude certain nodes
#SBATCH --cpus-per-task=2                   # specify cpu per task otherwise 8 per task
#SBATCH --mem=20G                           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=./slurm_log/slurm_%j.out          # File to which STDOUT will be written
#SBATCH --error=./slurm_log/slurm_%j.err           # File to which STDERR will be written
#SBATCH --mail-type=END                     # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=zhenzhong.xiao@uni-tuebingen.de  # Email to which notifications will be sent
#SBATCH --time=0-10:00            # Runtime in D-HH:MM


set -o errexit

# print info about current job
echo "---------- JOB INFOS ------------"
pwd
scontrol show job $SLURM_JOB_ID 
echo "node:"
hostname
echo -e "---------------------------------\n"


# Run code with values specified in task array
echo "-------- PYTHON OUTPUT ----------"

echo "RUN Script"

# options:
# --config: "mnist_blackbox_implicit.yaml", "cifar10.yaml", "cifar100.yaml"
# --seed: [10094, 20058, 27026, 48495, 65800]
# --model: ['CNN3', 'CNN6', 'CNN9', 'CNN15', 'MLP']
# --experiment: ['SGD', 'Student', 'Baseline']
# --collect: collect results


python train_slurm.py \
--config mnist_blackbox_implicit.yaml \
--collect

echo "---------------------------------"