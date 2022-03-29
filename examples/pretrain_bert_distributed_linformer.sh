#!/bin/bash

GPUS_PER_NODE=$TENSOR_PARALLEL_SIZE
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=29100
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=../data/bert/my-bert_text_sentence
CHECKPOINT_PATH=checkpoints/my-bert_checkpoints
VOCAB_PATH=../vocab/bert-large-uncased-vocab.txt

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_bert.py \
       --tensor-model-parallel-size $TENSOR_PARALLEL_SIZE \
       --linformer-k $LINFORMER_K \
       --share-heads \
       --num-layers $NUM_LAYERS \
       --hidden-size $HIDDEN_SIZE \
       --num-attention-heads $NUM_HEADS \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOBAL_BATCH_SIZE \
       --seq-length $SEQ_LENGTH \
       --max-position-embeddings $SEQ_LENGTH \
       --train-iters $TRAIN_ITERS \
       --lr-decay-iters 990000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_PATH \
       --data-impl mmap \
       --split 949,50,1 \
       --lr 0.0001 \
       --min-lr 0.00001 \
       --lr-decay-style linear \
       --lr-warmup-fraction .01 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --log-interval 100 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16 \
       --bert-no-binary-head \
       --exp

rm -rf ./checkpoints/*