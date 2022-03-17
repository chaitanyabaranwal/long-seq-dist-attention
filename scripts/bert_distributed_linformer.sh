#!/bin/bash

np=${1:-"1"}
pipeline=${2:-"1"}
tensor=${3:-"4"}
seq=${4:-"512"}
mic_bs=${5:-"8"}
glb_bs=${6:-"8"}
linformer_k=${7:-"256"}
layer=${8:-"12"}
hidden=${9:-"768"}
heads=${10:-"12"}
iters=${11:-"50000"}

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
export LINFORMER_K=$linformer_k

srun bash ./examples/pretrain_bert_distributed_linformer.sh $ADDR 29500
