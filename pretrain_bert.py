# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain BERT"""

import torch
import torch.nn.functional as F
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import mpu
from megatron.data.dataset_utils import build_train_valid_test_datasets
from megatron.model import BertModel
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group
from megatron.p2p_communication import ring_forward


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building BERT model ...')

    args = get_args()
    num_tokentypes = 2 if args.bert_binary_head else 0
    model = BertModel(
        num_tokentypes=num_tokentypes,
        add_binary_head=args.bert_binary_head,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process)

    return model


def get_batch(data_iterator):
    """Build the batch."""

    # Items and their type.
    keys = ['text', 'types', 'labels', 'is_random', 'loss_mask', 'padding_mask']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    # NCCL does not support scatter yet
    # data_b = mpu.scatter_data(keys, data, datatype)
    # unpack
    # data_b = mpu.broadcast_data(keys, data, datatype)
    # tokens = data_b['text'].long()
    # types = data_b['types'].long()
    # sentence_order = data_b['is_random'].long()
    # loss_mask = data_b['loss_mask'].float()
    # lm_labels = data_b['labels'].long()
    # padding_mask = data_b['padding_mask'].long()

    ####################################
    # NOTE: for RingParallelAttention  #
    ####################################
    # unpack
    data_b = mpu.broadcast_data(keys, data, datatype)

    # # get tensor parallel local rank
    global_rank = torch.distributed.get_rank()
    local_world_size = mpu.get_tensor_model_parallel_world_size()
    local_rank = global_rank % local_world_size
    seq_length = data_b['text'].size(1)
    sub_seq_length = seq_length // local_world_size
    sub_seq_start = local_rank * sub_seq_length
    sub_seq_end = (local_rank+1) * sub_seq_length
    #
    # # Unpack.
    tokens = data_b['text'][:, sub_seq_start:sub_seq_end].long().clone()
    types = data_b['types'][:, sub_seq_start:sub_seq_end].long()
    sentence_order = data_b['is_random'].long()
    loss_mask = data_b['loss_mask'][:, sub_seq_start:sub_seq_end].float()
    lm_labels = data_b['labels'][:, sub_seq_start:sub_seq_end].long()
    padding_mask = data_b['padding_mask'].long()

    return tokens, types, sentence_order, loss_mask, lm_labels, padding_mask


def loss_func(loss_mask, sentence_order, output_tensor):
    lm_loss_, sop_logits = output_tensor

    if torch.isnan(sop_logits).any() or torch.isnan(lm_loss_).any():
        print(f'lm_loss_ is {lm_loss_}, sop_logits: {sop_logits}')

    lm_loss_ = lm_loss_.float()
    loss_mask = loss_mask.float()
    loss_mask_sum = loss_mask.sum()
    lm_loss = torch.sum(
        lm_loss_.view(-1) * loss_mask.reshape(-1))

    torch.distributed.all_reduce(
        loss_mask_sum,
        group=mpu.get_tensor_model_parallel_group()
    )

    torch.distributed.all_reduce(
        lm_loss,
        group=mpu.get_tensor_model_parallel_group()
    )

    lm_loss /= loss_mask_sum

    if torch.isnan(lm_loss).any():
        print(f'lm loss is {lm_loss}, loss_mask_sum: {loss_mask_sum}')

    if sop_logits is not None:
        sop_loss = F.cross_entropy(sop_logits.view(-1, 2).float(),
                                   sentence_order.view(-1),
                                   ignore_index=-1)
        sop_loss = sop_loss.float()
        loss = lm_loss + sop_loss

        averaged_losses = average_losses_across_data_parallel_group(
            [lm_loss, sop_loss])
        return loss, {'lm loss': averaged_losses[0],
                      'sop loss': averaged_losses[1]}

    else:
        loss = lm_loss
        averaged_losses = average_losses_across_data_parallel_group(
            [lm_loss])
        return loss, {'lm loss': averaged_losses[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    tokens, types, sentence_order, loss_mask, lm_labels, padding_mask = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    if not args.bert_binary_head:
        types = None

    # Forward pass through the model.
    output_tensor = model(tokens, padding_mask, tokentype_ids=types,
                          lm_labels=lm_labels)

    return output_tensor, partial(loss_func, loss_mask, sentence_order)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for BERT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        max_seq_length=args.seq_length,
        masked_lm_prob=args.mask_prob,
        short_seq_prob=args.short_seq_prob,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        binary_head=args.bert_binary_head)
    print_rank_0("> finished creating BERT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    pretrain(train_valid_test_datasets_provider, model_provider, forward_step,
             args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})
