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

"""Transformer."""
import math
import torch
import torch.nn.functional as F
import torch.distributed.nn

from megatron.mpu.initialize import get_tensor_model_parallel_rank
from megatron.mpu.initialize import get_tensor_model_parallel_group
from megatron import get_args
from megatron import mpu
from .module import MegatronModule
from megatron.model.enums import AttnMaskType, LayerType, AttnType
from megatron.model import LayerNorm
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.fused_bias_gelu import bias_gelu_impl
from megatron.model.utils import attention_mask_func, openai_gelu, erf_gelu

# flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     p: number of model parallel partitions
     np: n/p
     hp: h/p
     hn: h/n
     b: batch size
     s: sequence length
     l: number of layers
     lin_k: Linformer constant parameter
    Transformer takes input of size [s, b, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
"""

class ParallelMLP(MegatronModule):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    """

    def __init__(self, init_method, output_layer_init_method):
        super(ParallelMLP, self).__init__()
        args = get_args()

        # Project to 4h.
        self.dense_h_to_4h = mpu.ColumnParallelLinear(
            args.hidden_size,
            args.ffn_hidden_size,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True)

        self.bias_gelu_fusion = args.bias_gelu_fusion
        self.activation_func = F.gelu
        if args.openai_gelu:
            self.activation_func = openai_gelu
        elif args.onnx_safe:
            self.activation_func = erf_gelu

        # Project back to h.
        self.dense_4h_to_h = mpu.RowParallelLinear(
            args.ffn_hidden_size,
            args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True)


    def forward(self, hidden_states):

        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

        if self.bias_gelu_fusion:
             intermediate_parallel = \
                     bias_gelu_impl(intermediate_parallel, bias_parallel)
        else:
            intermediate_parallel = \
                self.activation_func(intermediate_parallel + bias_parallel)

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias


class ParallelAttention(MegatronModule):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(self, init_method,
                 output_layer_init_method, layer_number,
                 attention_type=AttnType.self_attn,
                 attn_mask_type=AttnMaskType.padding):
        super(ParallelAttention, self).__init__()
        args = get_args()
        self.fp16 = args.fp16
        self.bf16 = args.bf16

        self.apply_query_key_layer_scaling = args.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type

        projection_size = args.kv_channels * args.num_attention_heads

        # Per attention head and per partition values.
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(projection_size,
                                                    world_size)
        self.hidden_size_per_attention_head = mpu.divide(
            projection_size, args.num_attention_heads)
        self.num_attention_heads_per_partition = mpu.divide(
            args.num_attention_heads, world_size)

        # Strided linear layer.
        if attention_type == AttnType.self_attn:
            self.query_key_value = mpu.ColumnParallelLinear(
                args.hidden_size,
                3 * projection_size,
                gather_output=False,
                init_method=init_method)
        else:
            assert attention_type == AttnType.cross_attn
            self.query = mpu.ColumnParallelLinear(
                args.hidden_size,
                projection_size,
                gather_output=False,
                init_method=init_method)

            self.key_value = mpu.ColumnParallelLinear(
                args.hidden_size,
                2 * projection_size,
                gather_output=False,
                init_method=init_method)

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.fp16, self.bf16,
            self.attn_mask_type,
            args.masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(args.attention_dropout)

        # Output.
        self.dense = mpu.RowParallelLinear(
            projection_size,
            args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True)

    def forward(self, hidden_states, attention_mask, layer_past=None,
                get_key_value=False, encoder_output=None):
        # hidden_states: [sq, b, h]

        # =====================
        # Query, Key, and Value
        # =====================

        if self.attention_type == AttnType.self_attn:
            # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
            mixed_x_layer, _ = self.query_key_value(hidden_states)

            # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer,
             key_layer,
             value_layer) = mpu.split_tensor_along_last_dim(mixed_x_layer, 3)
        else:
            # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
            mixed_kv_layer, _ = self.key_value(encoder_output)

            # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
            new_tensor_shape = mixed_kv_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 2 * self.hidden_size_per_attention_head)
            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

            # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
            (key_layer,
             value_layer) = mpu.split_tensor_along_last_dim(mixed_kv_layer, 2)

            # Attention head [sq, b, h] --> [sq, b, hp]
            query_layer, _ = self.query(hidden_states)
            # [sq, b, hp] --> [sq, b, np, hn]
            new_tensor_shape = query_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 self.hidden_size_per_attention_head)
            query_layer = query_layer.view(*new_tensor_shape)

        # ==================================
        # Adjust key and value for inference
        # ==================================

        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer),
                                   key_layer), dim=0)
            value_layer = torch.cat((past_value.type_as(value_layer),
                                     value_layer), dim=0)
        if get_key_value:
            present = (key_layer, value_layer)

        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (query_layer.size(1),
                       query_layer.size(2),
                       query_layer.size(0),
                       key_layer.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2],
                                       output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3],
                                   output_size[0] * output_size[1], -1)

        # preallocting result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0]*output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=torch.cuda.current_device())

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),   # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0, alpha=(1.0/self.norm_factor))

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

        if get_key_value:
            with torch.no_grad():
                if layer_past is not None:
                    attention_mask = attention_mask[
                        ...,
                        attention_scores.size(3) - 1,
                        :attention_scores.size(3)].unsqueeze(2)
                else:
                    attention_mask = attention_mask[
                        ...,
                        :attention_scores.size(3),
                        :attention_scores.size(3)]

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores,
                                                  attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with mpu.get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1),
                       value_layer.size(2),
                       query_layer.size(0),
                       value_layer.size(3))

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0),
                                       output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                               output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)

        if get_key_value:
            output = [output, present]

        return output, bias

####################################
# NOTE: for RingParallelAttention  #
####################################
class RingParallelAttention(MegatronModule):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(self, init_method,
                 output_layer_init_method, layer_number,
                 attention_type=AttnType.self_attn,
                 attn_mask_type=AttnMaskType.padding):
        super(RingParallelAttention, self).__init__()
        args = get_args()
        self.fp16 = args.fp16
        self.bf16 = args.bf16

        self.apply_query_key_layer_scaling = args.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.hidden_size = args.hidden_size

        projection_size = args.kv_channels * args.num_attention_heads
        self.hidden_size_per_attention_head = mpu.divide(
            projection_size, args.num_attention_heads)
        self.num_attention_heads = args.num_attention_heads
        self.world_size = mpu.get_tensor_model_parallel_world_size()

        # Strided linear layer.
        if attention_type == AttnType.self_attn:
            self.query_key_value = mpu.Linear(
                args.hidden_size,
                3 * projection_size,
                init_method=init_method)
        else:
            raise NotImplementedError('cross-attention is not implemented for RingParallelAttention')
            # assert attention_type == AttnType.cross_attn
            # self.query = mpu.ColumnParallelLinear(
            #     args.hidden_size,
            #     projection_size,
            #     gather_output=False,
            #     init_method=init_method)
            #
            # self.key_value = mpu.ColumnParallelLinear(
            #     args.hidden_size,
            #     2 * projection_size,
            #     gather_output=False,
            #     init_method=init_method)

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.fp16, self.bf16,
            self.attn_mask_type,
            args.masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(args.attention_dropout)

        # Output.
        self.dense = mpu.Linear(
            projection_size,
            args.hidden_size,
            init_method=output_layer_init_method,
            skip_bias_add=True)

    def forward(self, hidden_states, attention_mask, layer_past=None,
                get_key_value=False, encoder_output=None):
        # hidden_states: [sq, b, h]

        # =====================
        # Query, Key, and Value
        # =====================

        if self.attention_type == AttnType.self_attn:
            # Attention heads [sq, b, h] --> [sq, b, (3 * hn * num_heads)]
            mixed_x_layer, _ = self.query_key_value(hidden_states)

            # [sq, b, num_heads, 3 * hn] --> 3 [sq, b, num_heads, hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                               (self.num_attention_heads,
                                3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)
            (query_layer,
             key_layer,
             value_layer) = mpu.split_tensor_along_last_dim(mixed_x_layer, 3)
        else:
            raise NotImplementedError('cross-attention is not implemented for RingParallelAttention')
            # # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
            # mixed_kv_layer, _ = self.key_value(encoder_output)
            #
            # # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
            # new_tensor_shape = mixed_kv_layer.size()[:-1] + \
            #     (self.num_attention_heads_per_partition,
            #      2 * self.hidden_size_per_attention_head)
            # mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)
            #
            # # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
            # (key_layer,
            #  value_layer) = mpu.split_tensor_along_last_dim(mixed_kv_layer, 2)
            #
            # # Attention head [sq, b, h] --> [sq, b, hp]
            # query_layer, _ = self.query(hidden_states)
            # # [sq, b, hp] --> [sq, b, np, hn]
            # new_tensor_shape = query_layer.size()[:-1] + \
            #     (self.num_attention_heads_per_partition,
            #      self.hidden_size_per_attention_head)
            # query_layer = query_layer.view(*new_tensor_shape)

        # ==================================
        # Adjust key and value for inference
        # ==================================

        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer),
                                   key_layer), dim=0)
            value_layer = torch.cat((past_value.type_as(value_layer),
                                     value_layer), dim=0)
        if get_key_value:
            present = (key_layer, value_layer)

        # ===================================
        # Raw attention scores. [b, num_heads, s, s]
        # ===================================

        # [b, num_heads, sq, sk]
        output_size = (query_layer.size(1),
                       query_layer.size(2),
                       query_layer.size(0),
                       key_layer.size(0) * self.world_size)

        # [sq, b, num_heads, hn] -> [sq, b * num_heads, hn]
        query_layer = query_layer.view(output_size[2],
                                       output_size[0] * output_size[1], -1)
        # [sk, b, num_heads, hn] -> [sk, b * num_heads, hn]
        key_layer = key_layer.view(key_layer.size(0),
                                   output_size[0] * output_size[1], -1)

        # # preallocting result tensor: [b * np, sq, sk]
        # matmul_result = torch.empty(
        #     output_size[0]*output_size[1],
        #     output_size[2],
        #     output_size[3],
        #     dtype=query_layer.dtype,
        #     device=torch.cuda.current_device())
        #
        # # Raw attention scores. [b * np, sq, sk]
        # matmul_result = torch.baddbmm(
        #     matmul_result,
        #     query_layer.transpose(0, 1),   # [b * np, sq, hn]
        #     key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
        #     beta=0.0, alpha=(1.0/self.norm_factor))

        # [b, sq, sk]
        attention_scores = mpu.RingQK.apply(
            query_layer.transpose(0, 1).contiguous(), # [b * num_heads, sq, hn]
            key_layer.transpose(0, 1).contiguous() # [b * num_heads, sk, hn]
        )
        attention_scores /= self.norm_factor

        # change view to [b, num_heads, sq, sk]
        attention_scores = attention_scores.view(*output_size)

        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

        # if get_key_value:
        #     with torch.no_grad():
        #         if layer_past is not None:
        #             attention_mask = attention_mask[
        #                 ...,
        #                 attention_scores.size(3) - 1,
        #                 :attention_scores.size(3)].unsqueeze(2)
        #         else:
        #             attention_mask = attention_mask[
        #                 ...,
        #                 :attention_scores.size(3),
        #                 :attention_scores.size(3)]

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, num_heads, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores,
                                                  attention_mask)

        # attention_scores = attention_scores.unsqueeze(1)
        # attention_scores = attention_scores + attention_mask
        # attention_probs = F.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with mpu.get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, num_heads, hn] --> [b, num_heads, sq, hn]

        # context layer shape: [b, num_heads, sq, hn]
        output_size = (value_layer.size(1),
                       value_layer.size(2),
                       query_layer.size(0),
                       value_layer.size(3))
        #
        # # change view [sk, b * num_heads, hn]
        value_layer = value_layer.contiguous().view(value_layer.size(0),
                                       output_size[0] * output_size[1], -1)

        # # change view [b * num_heads, sq, sk]
        attention_probs = attention_probs.view(attention_probs.size(0) * attention_probs.size(1),
                                               attention_probs.size(2),
                                               attention_probs.size(3))

        # matmul: [b*num_heads, sq, hn]
        # context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))
        context_layer = mpu.RingAV.apply(attention_probs, value_layer.transpose(0, 1).contiguous())

        # # change view [b, num_heads, sq, hn]
        context_layer = context_layer.view(*output_size)

        # # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_attention_head * self.num_attention_heads,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================
        # context_layer = context_layer.transpose(1, 0).contiguous()
        output, bias = self.dense(context_layer)

        if get_key_value:
            output = [output, present]

        return output, bias

#############################################
# NOTE: for LinformerRingParallelAttention  #
#############################################
class LinformerRingParallelAttention(MegatronModule):
    """
    Ring-parallel self attention layer abstract class, which is combined
    with the linear complexity Transformer Linformer to reduce attention
    computation complexity even further.

    Original paper can be found at https://arxiv.org/abs/2006.04768.

    The current implementation does not have layerwise sharing or key-value sharing
    but has headwise sharing. Alternative configurations can be implemented as future work.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(self, init_method,
                 output_layer_init_method,
                 linformer_layer_init_method,
                 layer_number,
                 attention_type=AttnType.self_attn,
                 attn_mask_type=AttnMaskType.padding):
        super(LinformerRingParallelAttention, self).__init__()
        args = get_args()
        self.fp16 = args.fp16
        self.bf16 = args.bf16

        self.apply_query_key_layer_scaling = args.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.hidden_size = args.hidden_size
        self.linformer_k = args.linformer_k
        self.share_heads = args.share_heads

        projection_size = args.kv_channels * args.num_attention_heads
        self.hidden_size_per_attention_head = mpu.divide(
            projection_size, args.num_attention_heads)
        self.num_attention_heads = args.num_attention_heads
        self.world_size = mpu.get_tensor_model_parallel_world_size()

        # Strided linear layer.
        if attention_type == AttnType.self_attn:
            self.query_key_value = mpu.Linear(
                args.hidden_size,
                3 * projection_size,
                init_method=init_method)
        else:
            raise NotImplementedError('cross-attention is not implemented for LinformerRingParallelAttention')
        
        # Add a projection matrix for original key matrix
        # This essentially will compute (K' = K^T * A), where K' is a (d * k) matrix.
        if args.share_heads:
            self.key_projection = mpu.Linear(
                args.sub_seq_length,
                self.linformer_k,
                bias=False,
                init_method=linformer_layer_init_method
            )
        else:
            self.key_projections = [
                mpu.Linear(
                    args.sub_seq_length,
                    self.linformer_k,
                    bias=False,
                    init_method=linformer_layer_init_method
                ) for _ in range(args.num_attention_heads)
            ]

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.fp16, self.bf16,
            self.attn_mask_type,
            args.masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff
        )

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(args.attention_dropout)

        # Add a projection matrix for original value matrix
        # This essentially will compute (V' = V^T * A), where V' is the transpose of the
        # (d * k) projected value matrix. Must remember to transpose the result from this.
        if args.share_heads:
            self.value_projection = mpu.Linear(
                args.sub_seq_length,
                self.linformer_k,
                bias=False,
                init_method=linformer_layer_init_method
            )
        else:
            self.value_projections = [
                mpu.Linear(
                    args.sub_seq_length,
                    self.linformer_k,
                    bias=False,
                    init_method=linformer_layer_init_method
                ) for _ in range(args.num_attention_heads)
            ]

        # Output.
        self.dense = mpu.Linear(
            projection_size,
            args.hidden_size,
            init_method=output_layer_init_method,
            skip_bias_add=True)

    def forward(self, hidden_states, attention_mask, layer_past=None,
                get_key_value=False, encoder_output=None):
        # hidden_states: [sq, b, h]

        # =====================
        # Query, Key, and Value
        # =====================

        if self.attention_type == AttnType.self_attn:
            # Attention heads [sq, b, h] --> [sq, b, (3 * hn * num_heads)]
            mixed_x_layer, _ = self.query_key_value(hidden_states)

            # [sq, b, num_heads, 3 * hn] --> 3 [sq, b, num_heads, hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                               (self.num_attention_heads,
                                3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)
            (query_layer,
             key_layer,
             value_layer) = mpu.split_tensor_along_last_dim(mixed_x_layer, 3)
        else:
            raise NotImplementedError('cross-attention is not implemented for RingParallelAttention')

        # ==================================
        # Adjust key and value for inference
        # ==================================

        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer),
                                   key_layer), dim=0)
            value_layer = torch.cat((past_value.type_as(value_layer),
                                     value_layer), dim=0)
        if get_key_value:
            present = (key_layer, value_layer)

        # =============================================
        # Apply attention mask to key and value
        # because Linformer mask mechanism is different
        # =============================================

        # [b, num_heads, sq, lin_k]
        output_size = (query_layer.size(1),
                       query_layer.size(2),
                       query_layer.size(0),
                       self.linformer_k)

        # [sk, b, num_heads, hn] -> [b, num_heads, sk, hn]
        key_layer = key_layer.view(output_size[0], output_size[1],
                                   key_layer.size(0), -1).contiguous()
        assert attention_mask.size(0) == key_layer.size(0) and attention_mask.size(2) == key_layer.size(2), \
            'Attention mask dimensions do not match key matrix dimensions'
        key_layer.masked_fill_(attention_mask, 0.0)

        # [sk, b, num_heads, hn] -> [b, num_heads, sk, hn]
        value_layer = value_layer.view(output_size[0], output_size[1],
                                   value_layer.size(0), -1).contiguous()
        assert attention_mask.size(0) == value_layer.size(0) and attention_mask.size(2) == value_layer.size(2), \
            'Attention mask dimensions do not match value matrix dimensions'
        value_layer.masked_fill_(attention_mask, 0.0)

        # ===================================
        # Raw attention scores. [b, num_heads, s, s]
        # ===================================

        # [sq, b, num_heads, hn] -> [sq, b * num_heads, hn]
        query_layer = query_layer.view(output_size[2],
                                       output_size[0] * output_size[1], -1)
        

        # Project key matrix into lower dimension, using projection layer
        if self.share_heads:
            # [b, num_heads, sk, hn] -> [b * num_heads, hn, sk]
            key_layer = key_layer.view(output_size[0] * output_size[1], key_layer.size(3), key_layer.size(2))
            # [b * num_heads, hn, sk] -> [b * num_heads, hn, lin_k]
            key_layer, _ = self.key_projection(key_layer)
        else:
            # [b, num_heads, sk, hn] -> [b, num_heads, hn, sk]
            key_layer = key_layer.view(output_size[0], output_size[1], key_layer.size(3), key_layer.size(2))
            # [b, num_heads, hn, sk] -> [b * num_heads, hn, lin_k]
            key_layer_chunks = torch.chunk(key_layer, self.num_attention_heads, dim=1)
            key_layer = torch.cat(
                tuple(self.key_projections[i](chunk)[0] for i, chunk in enumerate(key_layer_chunks)),
                dim=1
            )
            key_layer = key_layer.view(output_size[0] * output_size[1], key_layer.size(2), key_layer.size(3))

        # collect all the projected key matrices into one
        torch.distributed.nn.all_reduce(key_layer, group=get_tensor_model_parallel_group())

        # compute local QK^T
        # [b * num_heads, sq, lin_k]
        attention_scores = torch.matmul(query_layer.transpose(0, 1), key_layer)
        attention_scores /= self.norm_factor

        # change view to [b, num_heads, sq, lin_k]
        attention_scores = attention_scores.view(*output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, num_heads, sq, lin_k]
        attention_mask = torch.broadcast_to(
            attention_mask,
            (attention_mask.size(0), attention_mask.size(1), attention_mask.size(2), self.linformer_k)
        ).contiguous()
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with mpu.get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [b, num_heads, sk, hn] --> [b, num_heads, sq, hn]

        # context layer shape: [b, num_heads, sq, hn]
        output_size = (value_layer.size(0),
                       value_layer.size(1),
                       query_layer.size(0),
                       value_layer.size(3))

        # Project key matrix into lower dimension, using projection layer
        if self.share_heads:
            # [b, num_heads, sk, hn] -> [b * num_heads, hn, sk]
            value_layer = value_layer.view(output_size[0] * output_size[1], value_layer.size(3), value_layer.size(2))
            # [b * num_heads, hn, sk] -> [b * num_heads, hn, lin_k]
            value_layer, _ = self.value_projection(value_layer)
        else:
            # [b, num_heads, sk, hn] -> [b, num_heads, hn, sk]
            value_layer = value_layer.view(output_size[0], output_size[1], value_layer.size(3), value_layer.size(2))
            # [b, num_heads, hn, sk] -> [b * num_heads, hn, lin_k]
            value_layer_chunks = torch.chunk(value_layer, self.num_attention_heads, dim=1)
            value_layer = torch.cat(
                tuple(self.value_projections[i](chunk)[0] for i, chunk in enumerate(value_layer_chunks)),
                dim=1
            )
            value_layer = value_layer.view(output_size[0] * output_size[1], value_layer.size(2), value_layer.size(3))

        # # change view [b * num_heads, sq, lin_k]
        attention_probs = attention_probs.view(attention_probs.size(0) * attention_probs.size(1),
                                               attention_probs.size(2),
                                               attention_probs.size(3))

        # collect all the projected value matrices into one
        # [b * num_heads, hn, lin_k] -> [b * num_heads, lin_k, hn]
        value_layer = value_layer.transpose(2, 1)
        torch.distributed.nn.all_reduce(value_layer, group=get_tensor_model_parallel_group())

        # compute local AV
        # [b * num_heads, sq, hn]
        context_layer = torch.matmul(attention_probs, value_layer)

        # # change view [b, num_heads, sq, hn]
        context_layer = context_layer.view(*output_size)

        # # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_attention_head * self.num_attention_heads,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================
        # context_layer = context_layer.transpose(1, 0).contiguous()
        output, bias = self.dense(context_layer)

        if get_key_value:
            output = [output, present]

        return output, bias

###########################################
# NOTE: for BigBirdRingParallelAttention  #
###########################################
class BigBirdRingParallelAttention(MegatronModule):
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

    The current implementation does not do fine-grained communication where
    the Q,K,V blocks are communicated selectively between machines. Currently
    it uses the ring-communication mechanism, modifications can be explored later.
    
    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(self, init_method,
                 output_layer_init_method,
                 layer_number,
                 attention_type=AttnType.self_attn,
                 attn_mask_type=AttnMaskType.padding):
        super(BigBirdRingParallelAttention, self).__init__()
        args = get_args()
        self.fp16 = args.fp16
        self.bf16 = args.bf16

        self.apply_query_key_layer_scaling = args.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.hidden_size = args.hidden_size
        self.block_size = args.block_size
        self.sub_seq_length = args.sub_seq_length
        self.seq_length = args.seq_length

        projection_size = args.kv_channels * args.num_attention_heads
        self.hidden_size_per_attention_head = mpu.divide(
            projection_size, args.num_attention_heads)
        self.num_attention_heads = args.num_attention_heads
        self.world_size = mpu.get_tensor_model_parallel_world_size()

        # Strided linear layer.
        if attention_type == AttnType.self_attn:
            self.query_key_value = mpu.Linear(
                args.hidden_size,
                3 * projection_size,
                init_method=init_method)
        else:
            raise NotImplementedError('cross-attention is not implemented for RingParallelAttention')

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.fp16, self.bf16,
            self.attn_mask_type,
            args.masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(args.attention_dropout)

        # Output.
        self.dense = mpu.Linear(
            projection_size,
            args.hidden_size,
            init_method=output_layer_init_method,
            skip_bias_add=True)

    def forward(self, hidden_states, attention_mask, layer_past=None,
                get_key_value=False, encoder_output=None,
                random_mapping=None):
        # hidden_states: [sq, b, h]

        # =====================
        # Query, Key, and Value
        # =====================

        if self.attention_type == AttnType.self_attn:
            # since using BigBird, check if subsequence can be broken down into blocks
            assert hidden_states.size(0) == self.sub_seq_length, 'args.sub_seq_length does not match data'
            local_blocks = mpu.divide(self.sub_seq_length, self.block_size)

            # Attention heads [sq, b, h] --> [sq, b, (3 * hn * num_heads)]
            mixed_x_layer, _ = self.query_key_value(hidden_states)

            # [sq, b, num_heads, 3 * hn] --> 3 [sq, b, num_heads, hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                               (self.num_attention_heads,
                                3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)
            (query_layer,
             key_layer,
             value_layer) = mpu.split_tensor_along_last_dim(mixed_x_layer, 3)
        else:
            raise NotImplementedError('cross-attention is not implemented for RingParallelAttention')

        # ==================================
        # Adjust key and value for inference
        # ==================================

        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer),
                                   key_layer), dim=0)
            value_layer = torch.cat((past_value.type_as(value_layer),
                                     value_layer), dim=0)
        if get_key_value:
            present = (key_layer, value_layer)

        # =========================================
        # Reshape query, key and value into blocks. 
        # [b, num_heads, local_blocks, block_size, h]
        # =========================================

        first_product_size = (
            query_layer.size(1),
            query_layer.size(2),
            self.block_size,
            self.seq_length
        )
        inner_product_size = (
            query_layer.size(1),
            query_layer.size(2),
            local_blocks * self.block_size,
            7 * self.block_size
        )

        # [sq, b, num_heads, hn] -> [b * num_heads, local_blocks, block_size, hn]
        query_layer = query_layer.view(inner_product_size[0] * inner_product_size[1], local_blocks, 
                                                    self.block_size, query_layer.size(3))
        key_layer = key_layer.view(inner_product_size[0] * inner_product_size[1], local_blocks, 
                                                    self.block_size, key_layer.size(3))

        # =========================================================================
        # Raw sparse attention scores.
        # First/last product: [b, num_heads, block_size, s]
        # Innter product: [b, num_heads, local_blocks * block_size, 5 * block_size]
        # =========================================================================

        first_product, inner_product, last_product = mpu.BigBirdRingQK.apply(
            query_layer.contiguous(), # [b * num_heads, local_blocks, block_size, hn]
            key_layer.contiguous(), # [b * num_heads, local_blocks, block_size, hn]
            random_mapping
        )
        if first_product is not None:
            first_product /= self.norm_factor
        inner_product /= self.norm_factor
        if last_product is not None:
            last_product /= self.norm_factor

        # change view to [b, num_heads, *, *]
        if first_product is not None:
            first_product = first_product.view(*first_product_size)
        if last_product is not None:
            last_product = last_product.view(*first_product_size)
        inner_product = inner_product.view(*inner_product_size)

        # =========================================================
        # Create attention mask the different attention components.
        # =========================================================

        first_product_mask = attention_mask[:, :, 0:self.block_size].view(
            attention_mask.size(0),
            attention_mask.size(1),
            self.block_size,
            self.seq_length
        ) if first_product is not None else None
        last_product_mask = attention_mask[:, :, ((local_blocks - 1) * self.block_size):(local_blocks * self.block_size)].view(
            attention_mask.size(0),
            attention_mask.size(1),
            self.block_size,
            self.seq_length
        ) if last_product is not None else None
        attention_mask = self.get_attention_mask(attention_mask, local_blocks, self.seq_length // self.block_size, self.block_size, random_mapping)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, num_heads, sq, sk]
        if first_product is not None:
            first_product = self.scale_mask_softmax(first_product, first_product_mask)
        if last_product is not None:
            last_product = self.scale_mask_softmax(last_product, last_product_mask)
        inner_product = self.scale_mask_softmax(inner_product, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with mpu.get_cuda_rng_tracker().fork():
            if first_product is not None:
                first_product = self.attention_dropout(first_product)
            if last_product is not None:
                last_product = self.attention_dropout(last_product)
            inner_product = self.attention_dropout(inner_product)

        # ============================================================
        # Context layer. [b, num_heads, local_blocks * block_size, hn]
        # ============================================================

        output_size = (
            value_layer.size(1),
            value_layer.size(2),
            local_blocks * self.block_size,
            value_layer.size(3)
        )

        # [sq, b, num_heads, hn] -> [b * num_heads, local_blocks, block_size, hn]
        value_layer = value_layer.view(output_size[0] * output_size[1], local_blocks,
                                            self.block_size, output_size[3])

        # change view of attention scores [b * num_heads, blocks, block_size, block_size]
        if first_product is not None:
            first_product = first_product.view(
                first_product.size(0) * first_product.size(1),
                self.seq_length // self.block_size,
                first_product.size(2),
                first_product.size(2)
            )
        if last_product is not None:
            last_product = last_product.view(
                last_product.size(0) * last_product.size(1),
                self.seq_length // self.block_size,
                last_product.size(2),
                last_product.size(2)
            )
        inner_product = inner_product.view(
            inner_product.size(0) * inner_product.size(1),
            local_blocks,
            self.block_size,
            inner_product.size(3)
        )

        # matmul: [b * num_heads, local_blocks, block_size, hn]
        context_layer = mpu.BigBirdRingAV.apply(
            first_product, inner_product, last_product,
            value_layer.contiguous(),
            random_mapping
        )

        # # change view [b, num_heads, sq, hn]
        context_layer = context_layer.view(*output_size)

        # # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_attention_head * self.num_attention_heads,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================
        # context_layer = context_layer.transpose(1, 0).contiguous()
        output, bias = self.dense(context_layer)

        if get_key_value:
            output = [output, present]

        return output, bias
    
    def get_attention_mask(self, attention_mask, local_blocks, total_blocks, block_size, random_mapping):
        # reshape attention mask to get diagonal
        band_shape = (attention_mask.size(0), attention_mask.size(1), local_blocks * block_size, block_size)
        attention_mask_reshape = attention_mask.view(
            attention_mask.size(0),
            attention_mask.size(1),
            local_blocks,
            total_blocks,
            block_size,
            block_size
        )

        rank = get_tensor_model_parallel_rank()
        start_block, end_block = (self.sub_seq_length * rank) // self.block_size, ((self.sub_seq_length * (rank + 1)) // self.block_size) - 1

        first_band = torch.diagonal(torch.roll(attention_mask_reshape, 1, 3)[:, :, :, start_block:(end_block + 1)], dim1=2, dim2=3).reshape(*band_shape)
        second_band = torch.diagonal(attention_mask_reshape[:, :, :, start_block:(end_block + 1)], dim1=2, dim2=3).reshape(*band_shape)
        third_band = torch.diagonal(torch.roll(attention_mask_reshape, -1, 3)[:, :, :, start_block:(end_block + 1)], dim1=2, dim2=3).reshape(*band_shape)

        # Get random mask mapping
        random_mask = torch.empty(
            attention_mask.size(0),
            attention_mask.size(1),
            attention_mask.size(2),
            2 * block_size,
            dtype=attention_mask.dtype,
            device=torch.cuda.current_device()
        )
        for i in range(local_blocks):
            random_mask[:, :, (i * block_size):((i + 1) * block_size), :block_size] = \
                attention_mask[:, :, (i * block_size):((i + 1) * block_size), (random_mapping[i][0] * block_size):((random_mapping[i][0] + 1) * block_size)]
            random_mask[:, :, (i * block_size):((i + 1) * block_size), block_size:(2 * block_size)] = \
                attention_mask[:, :, (i * block_size):((i + 1) * block_size), (random_mapping[i][1] * block_size):((random_mapping[i][1] + 1) * block_size)]

        return torch.cat((
            first_band, 
            second_band, 
            third_band, 
            attention_mask[:, :,:, :block_size], 
            attention_mask[:, :, :, (-1 * block_size):],
            random_mask),
            dim=3
        )

####################################
# NOTE: for RingParallelAttention  #
####################################
class MLP(MegatronModule):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    """

    def __init__(self, init_method, output_layer_init_method):
        super(MLP, self).__init__()
        args = get_args()

        # Project to 4h.
        self.dense_h_to_4h = mpu.Linear(
            args.hidden_size,
            args.ffn_hidden_size,
            init_method=init_method,
            skip_bias_add=True)

        self.bias_gelu_fusion = args.bias_gelu_fusion
        self.activation_func = F.gelu
        if args.openai_gelu:
            self.activation_func = openai_gelu
        elif args.onnx_safe:
            self.activation_func = erf_gelu

        # Project back to h.
        self.dense_4h_to_h = mpu.Linear(
            args.ffn_hidden_size,
            args.hidden_size,
            init_method=output_layer_init_method,
            skip_bias_add=True)


    def forward(self, hidden_states):

        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

        if self.bias_gelu_fusion:
             intermediate_parallel = \
                     bias_gelu_impl(intermediate_parallel, bias_parallel)
        else:
            intermediate_parallel = \
                self.activation_func(intermediate_parallel + bias_parallel)

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias

def bias_dropout_add(x, bias, residual, prob, training):
    # type: (Tensor, Tensor, Tensor, float, bool) -> Tensor
    out = torch.nn.functional.dropout(x + bias, p=prob, training=training)
    out = residual + out
    return out


def get_bias_dropout_add(training):
    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)
    return _bias_dropout_add


@torch.jit.script
def bias_dropout_add_fused_train(x, bias, residual, prob):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    return bias_dropout_add(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_add_fused_inference(x, bias, residual, prob):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    return bias_dropout_add(x, bias, residual, prob, False)


class ParallelTransformerLayer(MegatronModule):
    """A single transformer layer.

    Transformer layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(self, init_method, output_layer_init_method,
                 layer_number, layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding):
        args = get_args()

        super(ParallelTransformerLayer, self).__init__()
        self.layer_number = layer_number
        self.layer_type = layer_type

        self.apply_residual_connection_post_layernorm \
            = args.apply_residual_connection_post_layernorm

        self.bf16 = args.bf16
        self.fp32_residual_connection = args.fp32_residual_connection

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon)

        self.bigbird = args.bigbird

        ####################################
        # NOTE: for RingParallelAttention  #
        ####################################
        # Self attention.
        # Use Linformer based layer if applicable.
        if args.linformer_k:
            self.self_attention = LinformerRingParallelAttention(
                init_method,
                output_layer_init_method,
                init_method,
                layer_number,
                attention_type=AttnType.self_attn,
                attn_mask_type=self_attn_mask_type
            )
        # TODO (chai): Add BigBird only if sequence length > 1024
        elif args.bigbird:
            self.self_attention = BigBirdRingParallelAttention(
                init_method,
                output_layer_init_method,
                layer_number,
                attention_type=AttnType.self_attn,
                attn_mask_type=self_attn_mask_type
            )
        else:
            self.self_attention = RingParallelAttention(
                init_method,
                output_layer_init_method,
                layer_number,
                attention_type=AttnType.self_attn,
                attn_mask_type=self_attn_mask_type
            )

        self.hidden_dropout = args.hidden_dropout
        self.bias_dropout_fusion = args.bias_dropout_fusion

        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon)

        if self.layer_type == LayerType.decoder:
            self.inter_attention = ParallelAttention(
                init_method,
                output_layer_init_method,
                layer_number,
                attention_type=AttnType.cross_attn)
            # Layernorm on the attention output.
            self.post_inter_attention_layernorm = LayerNorm(
                args.hidden_size,
                eps=args.layernorm_epsilon)

        # MLP
        ####################################
        # NOTE: for RingParallelAttention  #
        ####################################
        self.mlp = MLP(init_method,
                               output_layer_init_method)
        # self.mlp = ParallelMLP(init_method,
        #                        output_layer_init_method)

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                layer_past=None, get_key_value=False,
                bigbird_random=None):
        # hidden_states: [b, s, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        if self.bigbird:
            assert bigbird_random is not None, 'Random attention needed in Big Bird'
            attention_output, attention_bias = \
                self.self_attention(layernorm_output,
                                    attention_mask,
                                    layer_past=layer_past,
                                    get_key_value=get_key_value,
                                    random_mapping=bigbird_random)
        else:
            attention_output, attention_bias = \
                self.self_attention(layernorm_output,
                                    attention_mask,
                                    layer_past=layer_past,
                                    get_key_value=get_key_value)

        if get_key_value:
            attention_output, presents = attention_output

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # jit scripting for a nn.module (with dropout) is not
        # trigerring the fusion kernel. For now, we use two
        # different nn.functional routines to account for varying
        # dropout semantics during training and inference phases.
        if self.bias_dropout_fusion:
            if self.training:
                bias_dropout_add_func = bias_dropout_add_fused_train
            else:
                bias_dropout_add_func = bias_dropout_add_fused_inference
        else:
            bias_dropout_add_func = get_bias_dropout_add(self.training)

        # re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            layernorm_input = bias_dropout_add_func(
                attention_output,
                attention_bias.expand_as(residual),
                residual,
                self.hidden_dropout)

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)


        if self.layer_type == LayerType.decoder:
            attention_output, attention_bias = \
                self.inter_attention(layernorm_output,
                                     enc_dec_attn_mask,
                                     encoder_output=encoder_output)
            # residual connection
            if self.apply_residual_connection_post_layernorm:
                residual = layernorm_output
            else:
                residual = layernorm_input

            # re-enable torch grad to enable fused optimization.
            with torch.enable_grad():
                layernorm_input = bias_dropout_add_func(
                    attention_output,
                    attention_bias.expand_as(residual),
                    residual,
                    self.hidden_dropout)

            # Layer norm post the decoder attention
            layernorm_output = self.post_inter_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output, mlp_bias = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        # re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            output = bias_dropout_add_func(
                mlp_output,
                mlp_bias.expand_as(residual),
                residual,
                self.hidden_dropout)

        if get_key_value:
            output = [output, presents]

        return output


class ParallelTransformer(MegatronModule):
    """Transformer class."""

    def __init__(self, init_method, output_layer_init_method,
                 layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 pre_process=True, post_process=True):
        super(ParallelTransformer, self).__init__()
        args = get_args()

        self.bf16 = args.bf16
        self.fp32_residual_connection = args.fp32_residual_connection
        self.pre_process = pre_process
        self.post_process = post_process
        self.input_tensor = None

        # Store activation checkpoiting flag.
        self.checkpoint_activations = args.checkpoint_activations
        self.checkpoint_num_layers = args.checkpoint_num_layers

        # Number of layers.
        assert args.num_layers % mpu.get_pipeline_model_parallel_world_size() == 0, \
            'num_layers must be divisible by pipeline_model_parallel_size'
        self.num_layers = args.num_layers // mpu.get_pipeline_model_parallel_world_size()

        # Transformer layers.
        def build_layer(layer_number):
            return ParallelTransformerLayer(
                init_method,
                output_layer_init_method,
                layer_number,
                layer_type=layer_type,
                self_attn_mask_type=self_attn_mask_type)
        if args.virtual_pipeline_model_parallel_size is not None:
            assert args.num_layers % args.virtual_pipeline_model_parallel_size == 0, \
                'num_layers_per_stage must be divisible by ' \
                'virtual_pipeline_model_parallel_size'
            # Number of layers in each model chunk is the number of layers in the stage,
            # divided by the number of model chunks in a stage.
            self.num_layers = self.num_layers // args.virtual_pipeline_model_parallel_size
            # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0]  [2]  [4]  [6]
            # Stage 1: [1]  [3]  [5]  [7]
            # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0, 1]  [4, 5]
            # Stage 1: [2, 3]  [6, 7]
            offset = mpu.get_virtual_pipeline_model_parallel_rank() * (
                args.num_layers // args.virtual_pipeline_model_parallel_size) + \
                (mpu.get_pipeline_model_parallel_rank() * self.num_layers)
        else:
            # Each stage gets a contiguous set of layers.
            offset = mpu.get_pipeline_model_parallel_rank() * self.num_layers

        self.layers = torch.nn.ModuleList(
            [build_layer(i + 1 + offset) for i in range(self.num_layers)])

        if self.post_process:
            # Final layer norm before output.
            self.final_layernorm = LayerNorm(
                args.hidden_size,
                eps=args.layernorm_epsilon)

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def _checkpointed_forward(self, hidden_states, attention_mask,
                              encoder_output, enc_dec_attn_mask,
                              bigbird_random=None):
        """Forward method with activation checkpointing."""
        def custom(start, end):
            def custom_forward(*inputs):
                x_ = inputs[0]
                attention_mask = inputs[1]
                encoder_output = inputs[2]
                enc_dec_attn_mask = inputs[3]
                for index in range(start, end):
                    layer = self._get_layer(index)
                    x_ = layer(x_, attention_mask, encoder_output, enc_dec_attn_mask, bigbird_random)
                return x_
            return custom_forward

        # Make sure memory is freed.
        mpu.reset_checkpointed_activations_memory_buffer()
        l = 0
        while l < self.num_layers:
            hidden_states = mpu.checkpoint(
                custom(l, l + self.checkpoint_num_layers),
                hidden_states, attention_mask, encoder_output, enc_dec_attn_mask)
            l += self.checkpoint_num_layers

        return hidden_states

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def forward(self, hidden_states, attention_mask, layer_past=None,
                get_key_value=False, encoder_output=None, enc_dec_attn_mask=None,
                bigbird_random=None):

        # Checks.
        if layer_past is not None:
            assert get_key_value, \
                'for not None values in layer_past, ' \
                'expected get_key_value to be set'
        if get_key_value:
            assert not self.checkpoint_activations, \
                'get_key_value does not work with ' \
                'activation checkpointing'

        if self.pre_process:
            # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
            # If the input flag for fp32 residual connection is set, convert for float.
            if self.fp32_residual_connection:
                hidden_states = hidden_states.transpose(0, 1).contiguous().float()
            # Otherwise, leave it as is.
            else:
                hidden_states = hidden_states.transpose(0, 1).contiguous()
        else:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        if encoder_output is not None:
             encoder_output = encoder_output.transpose(0, 1).contiguous()

        if self.checkpoint_activations:
            hidden_states = self._checkpointed_forward(hidden_states,
                                                       attention_mask,
                                                       encoder_output,
                                                       enc_dec_attn_mask,
                                                       bigbird_random)
        else:
            if get_key_value:
                presents = []
            for index in range(self.num_layers):
                layer = self._get_layer(index)
                past = None
                if layer_past is not None:
                    past = layer_past[index]
                hidden_states = layer(hidden_states,
                                      attention_mask,
                                      encoder_output=encoder_output,
                                      enc_dec_attn_mask=enc_dec_attn_mask,
                                      layer_past=past,
                                      get_key_value=get_key_value,
                                      bigbird_random=bigbird_random)
                if get_key_value:
                    hidden_states, present = hidden_states
                    presents.append(present)

        # Final layer norm.
        if self.post_process:
            # Reverting data format change [s b h] --> [b s h].
            hidden_states = hidden_states.transpose(0, 1).contiguous()
            output = self.final_layernorm(hidden_states)
        else:
            output = hidden_states
        if get_key_value:
            output = [output, presents]

        return output
