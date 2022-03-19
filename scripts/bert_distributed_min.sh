#!/bin/bash

module load daint-gpu
module load PyTorch

np=${1:-"1"}
pipeline=${2:-"1"}
tensor=${3:-"4"}
seq=${4:-"128"}
mic_bs=${5:-"8"}
glb_bs=${6:-"8"}
layer=${7:-"2"}
hidden=${8:-"256"}
heads=${9:-"8"}
iters=${10:-"110"}

python ./scripts/get_host_ip_addr.py > "./HOST"
ADDR=`cat ./HOST`
export MASTER_ADDR=$ADDR
export MASTER_PORT=29500
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

srun bash ./examples/pretrain_bert_distributed.sh
