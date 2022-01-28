#!/bin/bash

module load cuda/11.1.1-gcc-9.3.0
module load nccl

np=${1:-"2"}
pipeline=${2:-"2"}
tensor=${3:-"4"}
seq=${4:-"256"}
mic_bs=${5:-"2"}
glb_bs=${6:-"8"}
layer=${7:-"2"}
hidden=${8:-"256"}
heads=${9:-"8"}
iters=${10:-"10"}

#python ./scripts/get_host_ip_addr.py > "./HOST"
#ADDR=`cat ./HOST`
export MASTER_ADDR=localhost
export MASTER_PORT=6000
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

bash ./examples/pretrain_bert_distributed.sh

