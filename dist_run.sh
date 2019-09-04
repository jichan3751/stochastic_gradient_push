#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Provide two arg: rank world size"
    exit -1
fi

export SLURM_PROCID=$1
export SLURM_NTASKS=$2
export HOSTNAME=$3
# export HOSTNAME="127.0.0.1"

mkdir -p './ckpt/'

python -u gossip_sgd_mod.py \
    --batch_size 256 --lr 0.1 --num_dataloader_workers 4 \
    --num_epochs 1 --nesterov True --warmup True --push_sum False \
    --schedule 30 0.1 60 0.1 80 0.1 \
    --train_fast False --master_port 40100 \
    --tag 'AR-SGD-ETH' --print_freq 100 --verbose False \
    --graph_type -1 --all_reduce True --seed 1 \
    --network_interface_type 'ethernet' \
    --checkpoint_dir './ckpt/' \
    --dataset_dir ~/FF/datasets/cifar10

# python -u gossip_sgd.py \
#     --batch_size 256 --lr 0.1 --num_dataloader_workers 16 \
#     --num_epochs 1 --nesterov True --warmup True --push_sum False \
#     --graph_type 1 --schedule 30 0.1 60 0.1 80 0.1 \
#     --train_fast False --master_port 40100 \
#     --tag 'DPSGD_ETH' --print_freq 100 --verbose False \
#     --all_reduce False --seed 1 \
#     --network_interface_type 'ethernet' \
#     --checkpoint_dir './ckpt/' \
#     --dataset_dir ~/FF/datasets/cifar10

# python -u gossip_sgd.py \
#     --batch_size 256 --lr 0.1 --num_dataloader_workers 16 \
#     --num_epochs 1 --nesterov True --warmup True --push_sum True \
#     --graph_type 0 --schedule 30 0.1 60 0.1 80 0.1 \
#     --train_fast False --master_port 40100 \
#     --tag 'SGP_ETH' --print_freq 100 --verbose False \
#     --all_reduce False --seed 1 \
#     --network_interface_type 'ethernet' \
#     --checkpoint_dir './ckpt/' \
#     --dataset_dir ~/FF/datasets/cifar10
