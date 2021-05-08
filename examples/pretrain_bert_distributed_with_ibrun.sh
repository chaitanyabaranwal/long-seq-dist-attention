#!/bin/bash

GPUS_PER_NODE=4
# Change for multinode config
NODE_RANK=$PMI_RANK
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/scratch1/07825/franklee/projects/long-seq-transformer/data/wikipedia/my-bert_text_sentence

CHECKPOINT_PATH=/scratch1/07825/franklee/projects/long-seq-transformer/checkpoints/my-bert_checkpoints
VOCAB_PATH=/scratch1/07825/franklee/projects/long-seq-transformer/model/vocab.txt

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_bert.py \
       --tensor-model-parallel-size $TENSOR_PARALLEL_SIZE \
       --pipeline-model-parallel-size $PIPELINE_PARALLEL_SIZE \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 4 \
       --global-batch-size 16 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --train-iters 1000000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_PATH \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0005 \
       --lr-decay-style linear \
       --min-lr 1.0e-5 \
       --lr-decay-iters 990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 100 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16
#       --initial-loss-scale 65536 \


#       --min-loss-scale 0.1
#       --optimizer sgd
#       --fp16-lm-cross-entropy

rm -rf ../checkpoints/*