#!/bin/bash

# Change for multinode config
GPUS_PER_NODE=4
NODE_RANK=$PMI_RANK
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# Change for training config
DATA_PATH=<PATH_TO_BERT_DATASET>
CHECKPOINT_PATH=<PATH_TO_CHECKPOINT>
VOCAB_PATH=<PATH_TO_VOCAB_FILE>

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_bert.py \
       --tensor-model-parallel-size $TENSOR_PARALLEL_SIZE \
       --pipeline-model-parallel-size $PIPELINE_PARALLEL_SIZE \
       --num-layers $NUM_LAYERS \
       --hidden-size $HIDDEN_SIZE \
       --num-attention-heads $NUM_HEADS \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size  $GLOBAL_BATCH_SIZE \
       --seq-length $SEQ_LENGTH \
       --max-position-embeddings $SEQ_LENGTH \
       --train-iters 150 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_PATH \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --min-lr 1.0e-5 \
       --lr-decay-iters 990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16