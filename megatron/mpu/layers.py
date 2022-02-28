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


# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch


import math

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from .initialize import get_tensor_model_parallel_rank
from .initialize import get_tensor_model_parallel_world_size
from .initialize import get_tensor_model_parallel_group
from .mappings import copy_to_tensor_model_parallel_region
from .mappings import gather_from_tensor_model_parallel_region
from .mappings import reduce_from_tensor_model_parallel_region
from .mappings import scatter_to_tensor_model_parallel_region
from .random import get_cuda_rng_tracker
from .utils import divide
from .utils import split_tensor_along_last_dim
from .utils import VocabUtility
from megatron import get_args
from megatron.p2p_communication import ring_forward



_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {'tensor_model_parallel': False,
                                      'partition_dim': -1,
                                      'partition_stride': 1}


def param_is_not_tensor_parallel_duplicate(param):
    return (hasattr(param, 'tensor_model_parallel') and
            param.tensor_model_parallel) or (
                get_tensor_model_parallel_rank() == 0)


def set_tensor_model_parallel_attributes(tensor, is_parallel, dim, stride):
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    # # Set the attributes.
    setattr(tensor, 'tensor_model_parallel', is_parallel)
    setattr(tensor, 'partition_dim', dim)
    setattr(tensor, 'partition_stride', stride)


def set_defaults_if_not_set_tensor_model_parallel_attributes(tensor):
    def maybe_set(attribute, value):
        if not hasattr(tensor, attribute):
            setattr(tensor, attribute, value)
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_set(attribute, _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS[attribute])


def copy_tensor_model_parallel_attributes(destination_tensor, source_tensor):
    def maybe_copy(attribute):
        if hasattr(source_tensor, attribute):
            setattr(destination_tensor, attribute,
                    getattr(source_tensor, attribute))
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_copy(attribute)


def _initialize_affine_weight_gpu(weight, init_method,
                                  partition_dim, stride=1):
    """Initialize affine weight for model parallel on GPU."""

    set_tensor_model_parallel_attributes(tensor=weight,
                                         is_parallel=True,
                                         dim=partition_dim,
                                         stride=stride)

    with get_cuda_rng_tracker().fork():
        init_method(weight)


def _initialize_affine_weight_cpu(weight, output_size, input_size,
                                  per_partition_size, partition_dim,
                                  init_method, stride=1,
                                  return_master_weight=False):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    set_tensor_model_parallel_attributes(tensor=weight,
                                         is_parallel=True,
                                         dim=partition_dim,
                                         stride=stride)

    # Initialize master weight
    master_weight = torch.empty(output_size, input_size,
                                dtype=torch.float,
                                requires_grad=False)
    init_method(master_weight)
    args = get_args()
    master_weight = master_weight.to(dtype=args.params_dtype)

    # Split and copy
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size,
                              dim=partition_dim)
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_world_size()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, num_embeddings, embedding_dim,
                 init_method=init.xavier_normal_):
        super(VocabParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Set the detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()
        # Divide the weight matrix along the vocaburaly dimension.
        self.vocab_start_index, self.vocab_end_index = \
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings, get_tensor_model_parallel_rank(),
                self.tensor_model_parallel_size)
        self.num_embeddings_per_partition = self.vocab_end_index - \
            self.vocab_start_index

        # Allocate weights and initialize.
        args = get_args()
        if args.use_cpu_initialization:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                dtype=args.params_dtype))
            _initialize_affine_weight_cpu(
                self.weight, self.num_embeddings, self.embedding_dim,
                self.num_embeddings_per_partition, 0, init_method)
        else:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                device=torch.cuda.current_device(), dtype=args.params_dtype))
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=0, stride=1)

    def forward(self, input_):
        if self.tensor_model_parallel_size > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | \
                         (input_ >= self.vocab_end_index)
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
            # Get the embeddings.
        output_parallel = F.embedding(masked_input, self.weight,
                                      self.padding_idx, self.max_norm,
                                      self.norm_type, self.scale_grad_by_freq,
                                      self.sparse)
        # Mask the output embedding.
        if self.tensor_model_parallel_size > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = reduce_from_tensor_model_parallel_region(output_parallel)
        return output


####################################
# NOTE: for RingParallelAttention  #
####################################
class VocabEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, num_embeddings, embedding_dim,
                 init_method=init.xavier_normal_):
        super(VocabEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Set the detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None

        # Allocate weights and initialize.
        args = get_args()
        if args.use_cpu_initialization:
            self.weight = Parameter(torch.empty(
                self.num_embeddings, self.embedding_dim,
                dtype=args.params_dtype))
            _initialize_affine_weight_cpu(
                self.weight, self.num_embeddings, self.embedding_dim,
                self.num_embeddings, 0, init_method)
        else:
            self.weight = Parameter(torch.empty(
                self.num_embeddings, self.embedding_dim,
                device=torch.cuda.current_device(), dtype=args.params_dtype))
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=0, stride=1)

    def forward(self, hidden_state):
        output = F.embedding(hidden_state, self.weight,
                              self.padding_idx, self.max_norm,
                              self.norm_type, self.scale_grad_by_freq,
                              self.sparse)
        return output


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gether on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip 
                       adding bias but instead return it.
    """

    def __init__(self, input_size, output_size, bias=True, gather_output=True,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_world_size()
        self.output_size_per_partition = divide(output_size, world_size)
        self.skip_bias_add = skip_bias_add

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        args = get_args()
        if args.use_cpu_initialization:
            self.weight = Parameter(torch.empty(self.output_size_per_partition,
                                                self.input_size,
                                                dtype=args.params_dtype))
            self.master_weight = _initialize_affine_weight_cpu(
                self.weight, self.output_size, self.input_size,
                self.output_size_per_partition, 0, init_method,
                stride=stride, return_master_weight=keep_master_weight_for_test)
        else:
            self.weight = Parameter(torch.empty(
                self.output_size_per_partition, self.input_size,
                device=torch.cuda.current_device(), dtype=args.params_dtype))
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=0, stride=stride)
            
        if bias:
            if args.use_cpu_initialization:
                self.bias = Parameter(torch.empty(
                    self.output_size_per_partition, dtype=args.params_dtype))
            else:
                self.bias = Parameter(torch.empty(
                    self.output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=args.params_dtype))
            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)



    def forward(self, input_):
        # Set up backprop all-reduce.
        input_parallel = copy_to_tensor_model_parallel_region(input_)
        # Matrix multiply.

        bias = self.bias if not self.skip_bias_add else None
        output_parallel = F.linear(input_parallel, self.weight, bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel 
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip 
                       adding bias but instead return it.
    """

    def __init__(self, input_size, output_size, bias=True,
                 input_is_parallel=False,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, world_size)
        self.skip_bias_add = skip_bias_add

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        args = get_args()
        if args.use_cpu_initialization:
            self.weight = Parameter(torch.empty(self.output_size,
                                                self.input_size_per_partition,
                                                dtype=args.params_dtype))
            self.master_weight = _initialize_affine_weight_cpu(
                self.weight, self.output_size, self.input_size,
                self.input_size_per_partition, 1, init_method,
                stride=stride, return_master_weight=keep_master_weight_for_test)
        else:
            self.weight = Parameter(torch.empty(
                self.output_size, self.input_size_per_partition,
                device=torch.cuda.current_device(), dtype=args.params_dtype))
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=1, stride=stride)
        if bias:
            if args.use_cpu_initialization:
                self.bias = Parameter(torch.empty(self.output_size,
                                                  dtype=args.params_dtype))
            else:
                self.bias = Parameter(torch.empty(
                    self.output_size, device=torch.cuda.current_device(),
                    dtype=args.params_dtype))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)



    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight)
        # All-reduce across all the partitions.
        output_ = reduce_from_tensor_model_parallel_region(output_parallel)
        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias


class Linear(torch.nn.Module):
    """
    Linear layer with no parallelism. This is used because all linear operations
    in sequence parallelism are not computed in a row or column parallel fashion.

    The linear layer is defined as Y = XA + b.

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.
    """

    def __init__(self, input_size, output_size, bias=True,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False):
        super(Linear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        args = get_args()
        if args.use_cpu_initialization:
            self.weight = Parameter(torch.empty(self.output_size,
                                                self.input_size,
                                                dtype=args.params_dtype))
            self.master_weight = _initialize_affine_weight_cpu(
                self.weight, self.output_size, self.input_size,
                self.output_size_per_partition, 0, init_method,
                stride=stride, return_master_weight=keep_master_weight_for_test)
        else:
            self.weight = Parameter(torch.empty(
                self.output_size, self.input_size,
                device=torch.cuda.current_device(), dtype=args.params_dtype))
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=0, stride=stride)

        if bias:
            if args.use_cpu_initialization:
                self.bias = Parameter(torch.empty(
                    self.output_size, dtype=args.params_dtype))
            else:
                self.bias = Parameter(torch.empty(
                    self.output_size,
                    device=torch.cuda.current_device(),
                    dtype=args.params_dtype))
            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

    def forward(self, input_):
        # Matrix multiply.
        bias = self.bias if not self.skip_bias_add else None
        output = F.linear(input_, self.weight, bias)
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias



####################################
# NOTE: for RingParallelAttention  #
####################################

def _calc_incoming_device_range(i, rank, world_size):
    args = get_args()

    device_of_incoming_k = (rank - i - 1) % world_size
    start_idx = args.sub_seq_length * device_of_incoming_k
    end_idx = args.sub_seq_length * (device_of_incoming_k + 1)
    return start_idx, end_idx

def _calc_current_device_range(rank):
    args = get_args()
    start_idx = args.sub_seq_length * rank
    end_idx = args.sub_seq_length * (rank + 1)
    return start_idx, end_idx

class RingQK(torch.autograd.Function):
    """
    Calculate QK^T in a ring-exchange style
    """

    @staticmethod
    def forward(ctx, sub_q, sub_k):
        # save tensor for backward
        ctx.save_for_backward(sub_q, sub_k)

        args = get_args()

        # create local segment of attention score
        attention_score = torch.empty(args.micro_batch_size * args.num_attention_heads, args.sub_seq_length, args.seq_length,
                                      dtype=sub_q.dtype,
                                      device=torch.cuda.current_device()
                                      )

        # compute local QK^T
        part_a = torch.matmul(sub_q, sub_k.transpose(2, 1))
        local_rank = get_tensor_model_parallel_rank()
        local_world_size = get_tensor_model_parallel_world_size()
        start_idx = local_rank * args.sub_seq_length
        end_idx = (local_rank + 1) * args.sub_seq_length
        attention_score[:, :, start_idx: end_idx] = part_a

        # compute QK^T in ring-all-reduce style
        for i in range(local_world_size - 1):
            sub_k = ring_forward(sub_k)
            start_idx, end_idx = _calc_incoming_device_range(i, local_rank, local_world_size)
            part_a = torch.matmul(sub_q, sub_k.transpose(2, 1))
            attention_score[:, :, start_idx:end_idx] = part_a

        return attention_score

    @staticmethod
    def backward(ctx, grad_output):
        args = get_args()
        sub_q, sub_k, = ctx.saved_tensors
        local_rank = get_tensor_model_parallel_rank()
        local_world_size = get_tensor_model_parallel_world_size()

        # calculate gradient of sub_k
        grad_k = torch.matmul(
            grad_output.transpose(2, 1),
            sub_q
        )
        torch.distributed.all_reduce(grad_k, group=get_tensor_model_parallel_group())
        grad_k = grad_k[:, local_rank * args.sub_seq_length: (local_rank + 1) * args.sub_seq_length]
        grad_k /= local_world_size

        # calculate gradient for sub_q
        grad_q = torch.zeros_like(sub_q,
                                  dtype=sub_q.dtype,
                                  device=torch.cuda.current_device(),)

        # compute with local sub_k
        start_idx, end_idx = _calc_current_device_range(local_rank)
        grad_q += torch.matmul(grad_output[:, :, start_idx:end_idx], sub_k)

        # compute QK^T in ring-all-reduce style
        for i in range(local_world_size - 1):
            sub_k = ring_forward(sub_k)
            start_idx, end_idx = _calc_incoming_device_range(i, local_rank, local_world_size)
            grad_q += torch.matmul(grad_output[:, :, start_idx: end_idx], sub_k)

        grad_q /= local_world_size

        return grad_q, grad_k


class RingAV(torch.autograd.Function):

    @staticmethod
    def forward(ctx, attention_score, sub_v):
        args = get_args()
        local_rank = get_tensor_model_parallel_rank()
        local_world_size = get_tensor_model_parallel_world_size()
        local_start_idx, local_end_idx = _calc_current_device_range(local_rank)

        sub_attention_result = torch.zeros(args.micro_batch_size * args.num_attention_heads,
                                           args.sub_seq_length,
                                           args.hidden_size // args.num_attention_heads,
                                           device=torch.cuda.current_device(),
                                           dtype=attention_score.dtype)

        # save tensors for backward
        ctx.save_for_backward(attention_score, sub_v)

        # compute local AV
        part_av = torch.matmul(attention_score[:, :, local_start_idx:local_end_idx], sub_v)
        sub_attention_result += part_av

        # compute AV in ring - all - reduce style
        for i in range(local_world_size - 1):
            sub_v = ring_forward(sub_v)
            start_idx, end_idx = _calc_incoming_device_range(i, local_rank, local_world_size)

            # compute QK^T
            part_av = torch.matmul(attention_score[:, :, start_idx:end_idx], sub_v)
            sub_attention_result += part_av
        return sub_attention_result

    @staticmethod
    def backward(ctx, grad_output):
        local_rank = get_tensor_model_parallel_rank()
        local_world_size = get_tensor_model_parallel_world_size()
        local_start_idx, local_end_idx = _calc_current_device_range(local_rank)
        attention_scores, sub_v = ctx.saved_tensors

        # calculate gradient of v
        grad_v = torch.matmul(
            attention_scores.transpose(2, 1),
            grad_output
        )
        torch.distributed.all_reduce(grad_v, group=get_tensor_model_parallel_group())
        grad_v = grad_v[:, local_start_idx:local_end_idx]
        grad_v /= local_world_size

        # calculate gradient for attention score
        grad_attention_score = torch.zeros_like(attention_scores,
                                                dtype=grad_output.dtype,
                                                device=torch.cuda.current_device())

        # compute with local sub_k
        grad_attention_score[:, :, local_start_idx:local_end_idx] += torch.matmul(
            grad_output,
            sub_v.transpose(2, 1))

        # compute QK^T in ring-all-reduce style
        for i in range(local_world_size - 1):
            sub_v = ring_forward(sub_v)
            start_idx, end_idx = _calc_incoming_device_range(i, local_rank, local_world_size)

            # compute grad_q
            grad_attention_score[:, :, start_idx:end_idx] += torch.matmul(
                grad_output,
                sub_v.transpose(2, 1))

        return grad_attention_score, grad_v

###########################################
# NOTE: for BigBirdRingParallelAttention  #
###########################################

"""
Ring-parallel self attention layer abstract class, which is combined
with the linear complexity transformer Big Bird to reduce attention
computation complexity even further.

Original paper can be found at https://arxiv.org/abs/2007.14062. We implement
the BigBird-ITC variant where:
    
    global tokens: 2 x block_size
    window tokens: 3 x block_size
    random tokens: num_rand_tokens x block_size

The BigBird code has been adapted from the HuggingFace implementation, which can be
found at https://github.com/huggingface/transformers.

Formulas for calculating gradient of a matrix
C = AB => grad_A = grad_C * B^T, grad_B = A^T * grad_C
C = A^T * B => grad_A = B * (grad_C)^T, grad_B = A * grad_C
C = A * B^T => grad_A = grad_C * B, grad_B = (grad_C)^T * A 
"""

def _calc_incoming_device_block_range(i, rank, world_size):
    args = get_args()
    
    start_idx, end_idx = _calc_incoming_device_range(i, rank, world_size)
    start_block = start_idx // args.block_size
    end_block = end_idx // args.block_size
    return start_block, end_block

def _calc_current_device_block_range(rank):
    args = get_args()
    
    start_idx, end_idx = _calc_current_device_range(rank)
    start_block = start_idx // args.block_size
    end_block = end_idx // args.block_size
    return start_block, end_block

class BigBirdRingQK(torch.autograd.Function):
    """
    Calculates the sparse QK^T in a ring-exchange style.
    The resultant attention matrix is a collection of blocks of attention, which
    will be selectively multiplied to the block V matrix to get the final output.
    """

    @staticmethod
    def forward(ctx, sub_block_q, sub_block_k):
        # Get arguments
        args = get_args()
        local_rank = get_tensor_model_parallel_rank()
        local_world_size = get_tensor_model_parallel_world_size()
        total_blocks = args.seq_length // args.block_size
        local_blocks = args.sub_seq_length // args.block_size
        cur_start_block, cur_end_block = _calc_current_device_block_range(local_rank)

        # create local segment of attention score
        attention_score = torch.empty(
            args.micro_batch_size * args.num_attention_heads,
            local_blocks,
            args.block_size,
            3 * args.block_size,
            dtype=sub_block_q.dtype,
            device=torch.cuda.current_device()
        )
        # create left and right key segments and first and last query block respectively
        # this is needed to batch calculate the sliding window attention later
        first_q_left_block = torch.empty(
            args.micro_batch_size * args.num_attention_heads,
            1,
            args.block_size,
            args.hidden_size // args.num_attention_heads,
            dtype=sub_block_q.dtype,
            device=torch.cuda.current_device()
        )
        last_q_right_block = torch.empty(
            args.micro_batch_size * args.num_attention_heads,
            1,
            args.block_size,
            args.hidden_size // args.num_attention_heads,
            dtype=sub_block_q.dtype,
            device=torch.cuda.current_device()
        )
        sub_block_k = sub_block_k.transpose(2, 3)

        # left of first block and right of last block remaining, since not locally present
        # compute the remaining blocks using ring communication
        for i in range(local_world_size - 1):
            sub_block_k = ring_forward(sub_block_k)

            start_block, end_block = _calc_incoming_device_block_range(i, local_rank, local_world_size)

            if start_block == (cur_end_block + 1) % total_blocks:
                last_q_right_block = sub_block_k[:, 0, :, :]
            if end_block == (cur_start_block - 1) % total_blocks:
                first_q_left_block = sub_block_k[:, -1, :, :]
        
        sub_block_k = torch.cat((first_q_left_block, sub_block_k, last_q_right_block), dim=1)

        # save tensor for backward
        ctx.save_for_backward(sub_block_q, sub_block_k)

        # computer local sparse QK^T sliding window attention
        attention_score[:, :, :, 0:args.block_size] = torch.matmul(sub_block_q, sub_block_k[:, :-2, :, :])
        attention_score[:, :, :, args.block_size:(2 * args.block_size)] = torch.matmul(sub_block_q, sub_block_k[:, 1:-1, :, :])
        attention_score[:, :, :, (2 * args.block_size):(3 * args.block_size)] = torch.matmul(sub_block_q, sub_block_k[:, 2:, :, :])
        
        return attention_score
    
    @staticmethod
    def backward(ctx, grad_output):
        # get saved tensors
        sub_block_q, sub_block_k = ctx.saved_tensors

        # Get arguments
        args = get_args()
        local_rank = get_tensor_model_parallel_rank()
        local_world_size = get_tensor_model_parallel_world_size()
        total_blocks = args.seq_length // args.block_size
        local_blocks = args.sub_seq_length // args.block_size
        cur_start_block, cur_end_block = _calc_current_device_block_range(local_rank)

        # calculate gradients of sub_block_q
        grad_block_q = torch.matmul(grad_output[:, :, :, 0:args.block_size], sub_block_k[:, :-2, :, :])
        grad_block_q += torch.matmul(grad_output[:, :, :, args.block_size:(2 * args.block_size)], sub_block_k[:, 1:-1, :, :])
        grad_block_q += torch.matmul(grad_output[:, :, :, (2 * args.block_size):(3 * args.block_size)], sub_block_k[:, 2:, :, :])
        grad_block_q /= 3
        
        # calculate gradient of sub_block_k
        grad_block_k = torch.matmul(
            grad_output[:, :, :, args.block_size:(2 * args.block_size)].transpose(2, 3), 
            sub_block_q
        )

        # if more than one block, part of left and right attention blocks can be locally computed
        if local_blocks > 1:
            grad_block_k[:, :-1, :, :] += torch.matmul(
                grad_output[:, 1:, :, 0:args.block_size].transpose(2, 3), sub_block_q[:, 1:, :, :]
            )
            grad_block_k[:, 1:, :, :] += torch.matmul(
                grad_output[:, :-1, :, (2 * args.block_size):(3 * args.block_size)].transpose(2, 3), sub_block_q[:, :-1, :, :]
            )
        
        # compute the grad_block_q and grad_block_k blocks using ring communication
        first_q_block_grad_k = torch.matmul(
            grad_output[:, 0, :, 0:args.block_size].transpose(2, 3), sub_block_q[:, 0, :, :]
        )
        last_q_block_grad_k = torch.matmul(
            grad_output[:, -1, :, (2 * args.block_size):(3 * args.block_size)].transpose(2, 3), sub_block_q[:, -1, :, :]
        )
        for i in range(local_world_size - 1):
            first_q_block_grad_k = ring_forward(first_q_block_grad_k)
            last_q_block_grad_k = ring_forward(last_q_block_grad_k)

            start_block, end_block = _calc_incoming_device_block_range(i, local_rank, local_world_size)

            if start_block == (cur_end_block + 1) % total_blocks:
                grad_block_k[:, -1, :, :] += first_q_block_grad_k
            if end_block == (cur_start_block - 1) % total_blocks:
                grad_block_k[:, 0, :, :] += last_q_block_grad_k      

        # divide by 3 because in BigBird-ITC, sliding window attention is 3 blocks per query/key block
        grad_block_q /= 3
        grad_block_k /= 3

        return grad_block_q, grad_block_k
