#!/bin/bash

np=${1:-"1"}
pipeline=${2:-"1"}
tensor=${3:-"4"}
seq=${4:-"512"}
mic_bs=${5:-"64"}
glb_bs=${6:-"64"}
layer=${7:-"12"}
hidden=${8:-"768"}
heads=${9:-"12"}
iters=${10:-"50000"}

python ./scripts/get_host_ip_addr.py > "./HOST"
ADDR=`cat ./HOST`
export PIPELINE_PARALLEL_SIZE=$pipeline
export TENSOR_PARALLEL_SIZE=$tensor
export NNODES=$np
export SEQ_LENGTH=$seq
export MICRO_BATCH_SIZE=$mic_bs
export GLOBAL_BATCH_SIZE=$glb_bs
export NUM_LAYERS=$layer
export HIDDEN_SIZE=$hidden
export NUM_HEADS=$heads
export TRAIN_ITERS=$iters

srun bash ./examples/pretrain_bert.sh $ADDR 29500