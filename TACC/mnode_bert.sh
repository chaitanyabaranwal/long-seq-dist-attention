#!/bin/bash
module load tacc-singularity

np=${1:-"2"}
pipeline=${2:-"2"}
tensor=${3:-"4"}

python ./TACC/get_host_ip_addr.py > "./HOST"
ADDR=`cat ./HOST`
export IBRUN_TASKS_PER_NODE=1
export MASTER_ADDR=$ADDR
export MASTER_PORT=29500
export PIPELINE_PARALLEL_SIZE=$pipeline
export TENSOR_PARALLEL_SIZE=$tensor
export NNODES=$np

ibrun -np $np singularity exec --nv ../pytorch18.simg  bash ./examples/pretrain_bert_distributed_with_ibrun.sh