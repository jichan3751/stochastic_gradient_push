#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Provide 4 arg: rank wsize master_hostname task"
    exit -1
fi


export SLURM_PROCID=$1
export SLURM_NTASKS=$2
export HOSTNAME=$3
# export HOSTNAME="127.0.0.1"

TASK=$4

mkdir -p './ckpt/'

if [ $TASK == 0 ]; then
    echo "AR-SGD-ETH"
    time python -u gossip_sgd_mod.py \
        --batch_size 256 --lr 0.1 --num_dataloader_workers 4 \
        --num_epochs 2 --nesterov True --warmup True --push_sum False \
        --schedule 30 0.1 60 0.1 80 0.1 \
        --train_fast False --master_port 40100 \
        --tag 'AR-SGD-ETH' --print_freq 100 --verbose False \
        --graph_type -1 --all_reduce True --seed 1 \
        --network_interface_type 'ethernet' \
        --checkpoint_dir './ckpt/' \
        --dataset_dir '/data/datasets/imagenet12'

elif [ $TASK == 1 ]; then
    echo "DPSGD_ETH"
    time python -u gossip_sgd_mod.py \
        --batch_size 256 --lr 0.1 --num_dataloader_workers 4 \
        --num_epochs 2 --nesterov True --warmup True --push_sum False \
        --graph_type 1 --schedule 30 0.1 60 0.1 80 0.1 \
        --train_fast False --master_port 40100 \
        --tag 'DPSGD_ETH' --print_freq 100 --verbose False \
        --all_reduce False --seed 1 \
        --network_interface_type 'ethernet' \
        --checkpoint_dir './ckpt/' \
        --dataset_dir '/data/datasets/imagenet12'

elif [ $TASK == 2 ]; then
    echo "SGP_ETH"
    time python -u gossip_sgd_mod.py \
        --batch_size 256 --lr 0.1 --num_dataloader_workers 4 \
        --num_epochs 2 --nesterov True --warmup True --push_sum True \
        --graph_type 0 --schedule 30 0.1 60 0.1 80 0.1 \
        --train_fast False --master_port 40100 \
        --tag 'SGP_ETH' --print_freq 100 --verbose False \
        --all_reduce False --seed 1 \
        --network_interface_type 'ethernet' \
        --checkpoint_dir './ckpt/' \
        --dataset_dir '/data/datasets/imagenet12'
else
    echo "wrong task number"
fi
