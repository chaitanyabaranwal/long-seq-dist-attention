import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

from functools import partial

# params
world_size = 4
batch_size = 4
num_heads = 4
seq_length = 128
hidden_size = 64
sub_seq_length = seq_length // world_size
block_size = 8

def ring_forward(tensor_send_next):
    global world_size

    ops = []

    current_rank = dist.get_rank()

    tensor_recv_prev = torch.empty_like(tensor_send_next, 
                                        requires_grad=True, 
                                        device=torch.cuda.current_device())

    # send to next rank
    send_next_op = dist.P2POp(
        dist.isend, tensor_send_next,
        (current_rank + 1) % world_size)
    ops.append(send_next_op)

    # receive from prev rank
    recv_prev_op = dist.P2POp(
        dist.irecv, tensor_recv_prev,
        (current_rank - 1) % world_size)
    ops.append(recv_prev_op)

    if current_rank % 2 == 0:
        ops = ops[::-1]

    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()

    # To protect against race condition when using batch_isend_irecv().
    torch.cuda.synchronize()

    return tensor_recv_prev

def _calc_incoming_device_range(i, rank, world_size):
    global sub_seq_length
    device_of_incoming_k = (rank - i - 1) % world_size
    start_idx = sub_seq_length * device_of_incoming_k
    end_idx = sub_seq_length * (device_of_incoming_k + 1)
    return start_idx, end_idx

def _calc_current_device_range(rank):
    global sub_seq_length
    start_idx = sub_seq_length * rank
    end_idx = sub_seq_length * (rank + 1)
    return start_idx, end_idx

def _calc_incoming_device_block_range(i, rank, world_size):
    global block_size
    start_idx, end_idx = _calc_incoming_device_range(i, rank, world_size)
    start_block = start_idx // block_size
    end_block = end_idx // block_size
    return start_block, end_block - 1

def _calc_current_device_block_range(rank):
    global block_size
    start_idx, end_idx = _calc_current_device_range(rank)
    start_block = start_idx // block_size
    end_block = end_idx // block_size
    return start_block, end_block - 1

def _calc_device_with_first_block():
    return 0

def _calc_device_with_last_block():
    global world_size
    return world_size - 1

def _calc_current_device_inner_product_blocks(rank, total_blocks):
    start_block, end_block = _calc_current_device_block_range(rank)
    inner_start_block = max(1, start_block)
    inner_end_block = min(total_blocks - 2, end_block)
    if inner_end_block < inner_start_block:
        return None
    return (inner_start_block, inner_end_block)

class BigBirdRingQK(torch.autograd.Function):
    """
    Calculates the sparse QK^T in a ring-exchange style.
    The resultant attention matrix is a collection of blocks of attention, which
    will be selectively multiplied to the block V matrix to get the final output.
    """

    @staticmethod
    def forward(ctx, sub_block_q, sub_block_k):
        # Get arguments
        global world_size, batch_size, seq_length, sub_seq_length, hidden_size, block_size, num_heads
        local_rank = dist.get_rank()
        total_blocks = seq_length // block_size
        local_blocks = sub_seq_length // block_size
        cur_start_block, cur_end_block = _calc_current_device_block_range(local_rank)

        # create local segment of attention scores
        first_product = torch.empty(
            batch_size * num_heads,
            total_blocks,
            block_size,
            block_size,
            dtype=sub_block_q.dtype,
            device=torch.cuda.current_device()    
        ) if cur_start_block <= 0 <= cur_end_block else None
        inner_product = torch.empty(
            batch_size * num_heads,
            local_blocks,
            block_size,
            5 * block_size,
            dtype=sub_block_q.dtype,
            device=torch.cuda.current_device()
        )
        last_product = torch.empty(
            batch_size * num_heads,
            total_blocks,
            block_size,
            block_size,
            dtype=sub_block_q.dtype,
            device=torch.cuda.current_device()    
        ) if cur_start_block <= total_blocks - 1 <= cur_end_block else None

        # create left and right key segments and first and last query block respectively
        # this is needed to batch calculate the sliding window attention later
        first_q_left_block = torch.empty(
            batch_size * num_heads,
            1,
            hidden_size,
            block_size,
            dtype=sub_block_q.dtype,
            device=torch.cuda.current_device()
        )
        last_q_right_block = torch.empty_like(first_q_left_block)
        sub_block_k = sub_block_k.transpose(2, 3)

        # first and last block pay attention from all blocks
        if first_product is not None:
            first_product[:, cur_start_block:(cur_end_block + 1)] = torch.matmul(
                sub_block_q[:, 0:1], sub_block_k
            )
            inner_product[:, :, :, (3 * block_size):(4 * block_size)] = torch.matmul(
                sub_block_q, sub_block_k[:, 0:1]
            )
        if last_product is not None:
            last_product[:, cur_start_block:(cur_end_block + 1)] = torch.matmul(
                sub_block_q[:, -1:], sub_block_k
            )
            inner_product[:, :, :, (4 * block_size):(5 * block_size)] = torch.matmul(
                sub_block_q, sub_block_k[:, -1:]
            )

        # left of first block and right of last block remaining, since not locally present
        # compute the remaining blocks using ring communication
        for i in range(world_size - 1):
            sub_block_k = ring_forward(sub_block_k)
            start_block, end_block = _calc_incoming_device_block_range(i, local_rank, world_size)

            # first and last blocks pay attention from all blocks
            if first_product is not None:
                first_product[:, start_block:(end_block + 1)] = torch.matmul(
                    sub_block_q[:, 0:1], sub_block_k
                )
            if last_product is not None:
                last_product[:, start_block:(end_block + 1)] = torch.matmul(
                    sub_block_q[:, -1:], sub_block_k
                )
            
            # first and last blocks get attention from all blocks
            if start_block == 0:
                inner_product[:, :, :, (3 * block_size):(4 * block_size)] = torch.matmul(
                    sub_block_q, sub_block_k[:, 0:1]
                )
            if end_block == total_blocks - 1:
                inner_product[:, :, :, (4 * block_size):(5 * block_size)] = torch.matmul(
                    sub_block_q, sub_block_k[:, -1:]
                )

            # gather any remaining blocks needed for sliding window attention
            if start_block == (cur_end_block + 1) % total_blocks:
                last_q_right_block = sub_block_k[:, 0:1]
            if end_block == (cur_start_block - 1) % total_blocks:
                first_q_left_block = sub_block_k[:, -1:]
        
        # get back original block
        sub_block_k = ring_forward(sub_block_k)
        # concatenate any extra key blocks needed for sliding window attention
        sub_block_k = torch.cat((first_q_left_block, sub_block_k, last_q_right_block), dim=1)

        # save tensor for backward
        ctx.save_for_backward(sub_block_q, sub_block_k)

        # computer QK^T sliding window attention
        # TODO (chai): Consider breaking down into parts to save memory
        inner_product[:, :, :, :(3 * block_size)] = torch.matmul(
            sub_block_q, 
            torch.cat((sub_block_k[:, :-2], sub_block_k[:, 1:-1], sub_block_k[:, 2:]), dim=3)
        )

        # apply mask to sliding window if first or second (or last or second last) block present
        inner_block_range = _calc_current_device_inner_product_blocks(local_rank, total_blocks)
        if first_product is not None and cur_start_block <= 1 <= cur_end_block:
            inner_product[:, 0].fill_(-10000.0)
            inner_product[:, 1, :, (3 * block_size):(4 * block_size)].fill_(-10000.0)
        elif first_product is not None:
            inner_product[:, 0].fill_(-10000.0)
        elif cur_start_block <= 1 <= cur_end_block:
            inner_product[:, 0, :, (3 * block_size):(4 * block_size)].fill_(-10000.0)
        if last_product is not None and cur_start_block <= total_blocks - 2 <= cur_end_block:
            inner_product[:, -1].fill_(-10000.0)
            inner_product[:, -2, :, (4 * block_size):(5 * block_size)].fill_(-10000.0)
        elif last_product is not None:
            inner_product[:, -1].fill_(-10000.0)
        elif cur_start_block <= total_blocks - 2 <= cur_end_block:
            inner_product[:, -1, :, (4 * block_size):(5 * block_size)].fill_(-10000.0)

        # return the first product, inner product, and last product
        return (first_product, inner_product, last_product)
    
    @staticmethod
    def backward(ctx, grad_output):
        # get gradients of different components
        (grad_first_product, grad_inner_product, grad_last_product) = grad_output

        # get saved tensors
        sub_block_q, sub_block_k = ctx.saved_tensors
        sub_block_k = sub_block_k.transpose(2, 3)

        # Get arguments
        global world_size, batch_size, seq_length, sub_seq_length, hidden_size, block_size, num_heads
        local_rank = dist.get_rank()
        total_blocks = seq_length // block_size
        local_blocks = sub_seq_length // block_size
        cur_start_block, cur_end_block = _calc_current_device_block_range(local_rank)

        # setup tensors for the gradients
        grad_block_q = torch.zeros_like(sub_block_q)
        grad_block_k = torch.zeros_like(grad_block_q, dtype=sub_block_k.dtype)

        # calculate local gradients of sub_block_q based on sliding window attention
        grad_block_q += torch.matmul(grad_inner_product[:, :, :, 0:block_size], sub_block_k[:, :-2])
        grad_block_q += torch.matmul(grad_inner_product[:, :, :, block_size:(2 * block_size)], sub_block_k[:, 1:-1])
        grad_block_q += torch.matmul(grad_inner_product[:, :, :, (2 * block_size):(3 * block_size)], sub_block_k[:, 2:])

        # calculate local gradients of sub_block_q based on global attention
        if grad_first_product is not None:
            grad_block_q[:, 0] += torch.sum(torch.matmul(grad_first_product[:, cur_start_block:(cur_end_block + 1)], sub_block_k[:, 1:-1]), dim=1)
            grad_block_q += torch.matmul(grad_inner_product[:, :, :, (3 * block_size):(4 * block_size)], sub_block_k[:, 1])
        if grad_last_product is not None:
            grad_block_q[:, -1] += torch.sum(torch.matmul(grad_last_product[:, cur_start_block:(cur_end_block + 1)], sub_block_k[:, 1:-1]), dim=1)
            grad_block_q += torch.matmul(grad_inner_product[:, :, :, (4 * block_size):(5 * block_size)], sub_block_k[:, -2])
        
        # calculate gradient of sub_block_k
        grad_block_k += torch.matmul(
            grad_inner_product[:, :, :, block_size:(2 * block_size)].transpose(2, 3), 
            sub_block_q
        )

        # if more than one block, part of left and right attention blocks can be locally computed
        if local_blocks > 1:
            grad_block_k[:, :-1] += torch.matmul(
                grad_inner_product[:, 1:, :, 0:block_size].transpose(2, 3), sub_block_q[:, 1:]
            )
            grad_block_k[:, 1:] += torch.matmul(
                grad_inner_product[:, :-1, :, (2 * block_size):(3 * block_size)].transpose(2, 3), sub_block_q[:, :-1]
            )
        
        # compute the grad_block_q and grad_block_k blocks using ring communication
        first_q_block_grad_k = torch.matmul(
            grad_inner_product[:, 0, :, 0:block_size].transpose(2, 3), sub_block_q[:, 0]
        )
        last_q_block_grad_k = torch.matmul(
            grad_inner_product[:, -1, :, (2 * block_size):(3 * block_size)].transpose(2, 3), sub_block_q[:, -1]
        )
        for i in range(world_size - 1):
            first_q_block_grad_k = ring_forward(first_q_block_grad_k)
            last_q_block_grad_k = ring_forward(last_q_block_grad_k)
            sub_block_k = ring_forward(sub_block_k)
            start_block, end_block = _calc_incoming_device_block_range(i, local_rank, world_size)

            # first and last blocks pay attention to all blocks
            if grad_first_product is not None:
                grad_block_q[:, 0] += torch.sum(torch.matmul(grad_first_product[:, start_block:(end_block + 1)], sub_block_k[:, 1:-1]), dim=1)
            if grad_last_product is not None:
                grad_block_q[:, -1] += torch.sum(torch.matmul(grad_last_product[:, start_block:(end_block + 1)], sub_block_k[:, 1:-1]), dim=1)

            # first and last block gets attention from all blocks
            if start_block == 0:
                grad_block_q += torch.matmul(grad_inner_product[:, :, :, (3 * block_size):(4 * block_size)], sub_block_k[:, 1])
            if end_block == total_blocks - 1:
                grad_block_q += torch.matmul(grad_inner_product[:, :, :, (4 * block_size):(5 * block_size)], sub_block_k[:, -2])

            if start_block == (cur_end_block + 1) % total_blocks:
                grad_block_k[:, -1] += first_q_block_grad_k
            if end_block == (cur_start_block - 1) % total_blocks:
                grad_block_k[:, 0] += last_q_block_grad_k

        # at this point, grad_block_k has the sliding window attention gradient computed
        # compute gradients from global attention by first and last query blocks
        grad_block_k_from_global = torch.zeros(
            batch_size * num_heads,
            total_blocks,
            block_size,
            hidden_size,
            dtype=grad_block_k.dtype,
            device=torch.cuda.current_device()
        )
        if grad_first_product is not None:
            grad_block_k_from_global += torch.matmul(grad_first_product.transpose(2, 3), sub_block_q[:, 0])
        if grad_last_product is not None:
            grad_block_k_from_global += torch.matmul(grad_last_product.transpose(2, 3), sub_block_q[:, -1])
        torch.distributed.all_reduce(grad_block_k_from_global, group=get_tensor_model_parallel_group())
        grad_block_k_from_global = grad_block_k_from_global[:, cur_start_block:(cur_end_block + 1)]
        grad_block_k += grad_block_k_from_global

        # compute gradients from global attention by first and last key blocks
        grad_block_k_from_global = torch.matmul(
            grad_inner_product[:, :, :, (3 * block_size):(4 * block_size)].transpose(2, 3),
            sub_block_q
        )
        grad_block_k_from_global = torch.sum(grad_block_k_from_global, dim=1)
        torch.distributed.reduce(grad_block_k_from_global, _calc_device_with_first_block(), group=get_tensor_model_parallel_group())
        if cur_start_block == 0:
            grad_block_k[:, 0] += grad_block_k_from_global
        
        grad_block_k_from_global = torch.matmul(
            grad_inner_product[:, :, :, (4 * block_size):(5 * block_size)].transpose(2, 3),
            sub_block_q
        )
        grad_block_k_from_global = torch.sum(grad_block_k_from_global, dim=1)
        torch.distributed.reduce(grad_block_k_from_global, _calc_device_with_last_block(), group=get_tensor_model_parallel_group())
        if cur_end_block == total_blocks - 1:
            grad_block_k[:, -1] += grad_block_k_from_global

        # scale the gradients accordingly
        inner_block_range = _calc_current_device_inner_product_blocks(local_rank, total_blocks)
        if grad_first_product is not None and cur_start_block <= 1 <= cur_end_block:
            grad_block_q[:, 0] /= world_size
            grad_block_k[:, 0] /= world_size
            grad_block_q[:, 1] /= 4
            grad_block_k[:, 1] /= 4
        elif grad_first_product is not None:
            grad_block_q[:, 0] /= world_size
            grad_block_k[:, 0] /= world_size
        elif cur_start_block <= 1 <= cur_end_block:
            grad_block_q[:, 0] /= 4
            grad_block_k[:, 0] /= 4
        if grad_last_product is not None and cur_start_block <= total_blocks - 2 <= cur_end_block:
            grad_block_q[:, -1] /= world_size
            grad_block_k[:, -1] /= world_size
            grad_block_q[:, -2] /= 4
            grad_block_k[:, -2] /= 4
        elif grad_last_product is not None:
            grad_block_q[:, -1] /= world_size
            grad_block_k[:, -1] /= world_size
        elif cur_start_block <= total_blocks - 2 <= cur_end_block:
            grad_block_q[:, -1] /= 4
            grad_block_k[:, -1] /= 4
        if inner_block_range:
            grad_block_q[:, inner_block_range[0]:(inner_block_range[1] + 1)] /= 5
            grad_block_k[:, inner_block_range[0]:(inner_block_range[1] + 1)] /= 5

        return grad_block_q, grad_block_k

class BigBirdRingAV(torch.autograd.Function):
    """
    Calculates the sparse AV in a ring-exchange style.
    The resultant attention matrix is a collection of blocks of output values.
    """

    @staticmethod
    def forward(ctx, first_product, inner_product, last_product, sub_block_v):
        # Get arguments
        global world_size, batch_size, seq_length, sub_seq_length, hidden_size, block_size, num_heads
        local_rank = dist.get_rank()
        total_blocks = seq_length // block_size
        local_blocks = sub_seq_length // block_size
        cur_start_idx, cur_end_idx = _calc_current_device_range(local_rank)
        cur_start_block, cur_end_block = _calc_current_device_block_range(local_rank)

        # create local segment of attention scores
        first_context = torch.zeros(
            batch_size * num_heads,
            1,
            block_size,
            hidden_size,
            dtype=first_product.dtype,
            device=torch.cuda.current_device()    
        ) if cur_start_block <= 0 <= cur_end_block else None
        inner_context = torch.zeros(
            batch_size * num_heads,
            local_blocks,
            block_size,
            hidden_size,
            dtype=inner_product.dtype,
            device=torch.cuda.current_device()    
        )
        last_context = torch.zeros(
            batch_size * num_heads,
            1,
            block_size,
            hidden_size,
            dtype=last_product.dtype,
            device=torch.cuda.current_device()    
        ) if cur_start_block <= total_blocks - 1 <= cur_end_block else None

        # create left and right value segments and first and last attention block respectively
        # this is needed to batch calculate the sliding window output later
        first_a_left_block = torch.empty(
            batch_size * num_heads,
            1,
            block_size,
            hidden_size,
            dtype=sub_block_v.dtype,
            device=torch.cuda.current_device()
        )
        last_a_right_block = torch.empty_like(first_a_left_block)

        # first and last attention block attend to all value blocks
        if first_context is not None:
            first_context += torch.sum(torch.matmul(first_product[:, cur_start_block:(cur_end_block + 1)], sub_block_v), dim=1, keepdims=True)
            inner_context += torch.matmul(inner_product[:, :, :, (3 * block_size):(4 * block_size)], sub_block_v[:, 0:1])
        if last_context is not None:
            last_context += torch.sum(torch.matmul(last_product[:, cur_start_block:(cur_end_block + 1)], sub_block_v), dim=1, keepdims=True)
            inner_context += torch.matmul(inner_product[:, :, :, (4 * block_size):(5 * block_size)], sub_block_v[:, -1:])
        
        # left of first attention block and right of last block remaining, since not locally present
        # compute the remaining blocks using ring communication
        for i in range(world_size - 1):
            sub_block_v = ring_forward(sub_block_v)
            start_block, end_block = _calc_incoming_device_block_range(i, local_rank, world_size)

            # first and last blocks pay attention to all blocks
            if first_context is not None:
                first_context += torch.sum(torch.matmul(first_product[:, start_block:(end_block + 1)], sub_block_v), dim=1, keepdims=True)
            if last_context is not None:
                last_context += torch.sum(torch.matmul(last_product[:, start_block:(end_block + 1)], sub_block_v), dim=1, keepdims=True)
            
            # first and last blocks get attention from all blocks
            if start_block == 0:
                inner_context += torch.matmul(inner_product[:, :, :, (3 * block_size):(4 * block_size)], sub_block_v[:, 0:1])
            if end_block == total_blocks - 1:
                inner_context += torch.matmul(inner_product[:, :, :, (4 * block_size):(5 * block_size)], sub_block_v[:, -1:])
            
            # gather any remaining value blocks needed for sliding window attention
            if start_block == (cur_end_block + 1) % total_blocks:
                last_a_right_block = sub_block_v[:, 0:1]
            if end_block == (cur_start_block - 1) % total_blocks:
                first_a_left_block = sub_block_v[:, -1:]
        
        # get back original block
        sub_block_v = ring_forward(sub_block_v)
        # concatenate any extra value blocks for sliding window attention
        sub_block_v = torch.cat((first_a_left_block, sub_block_v, last_a_right_block), dim=1)

        # save tensor for backward
        ctx.save_for_backward(first_product, inner_product, last_product, sub_block_v)

        # compute AV sliding window attention
        # TODO (chai): Consider breaking down into parts to save memory
        inner_context += torch.matmul(
            inner_product[:, :, :, :(3 * block_size)], 
            torch.cat((sub_block_v[:, :-2], sub_block_v[:, 1:-1], sub_block_v[:, 2:]), dim=2)
        )

        # concatenatate accordingly
        if first_context is not None and last_context is not None:
            context_layer = torch.cat((first_context, inner_context[:, 1:-1], last_context), dim=1)
        elif first_context is not None:
            context_layer = torch.cat((first_context, inner_context[:, 1:]), dim=1)
        elif last_context is not None:
            context_layer = torch.cat((inner_context[:, :-1], last_context), dim=1)
        else:
            context_layer = inner_context

        return context_layer

    @staticmethod
    def backward(ctx, grad_output):
        # get saved tensors
        first_product, inner_product, last_product, sub_block_v = ctx.saved_tensors

        # Get arguments
        global world_size, batch_size, seq_length, sub_seq_length, hidden_size, block_size, num_heads
        local_rank = dist.get_rank()
        total_blocks = seq_length // block_size
        local_blocks = sub_seq_length // block_size
        cur_start_block, cur_end_block = _calc_current_device_block_range(local_rank)

        # create gradient tensors for attention scores and value layer
        grad_first_product = torch.zeros_like(
            first_product, 
            dtype=grad_output.dtype, 
            device=torch.cuda.current_device()
        ) if first_product else None
        grad_last_product = torch.zeros_like(
            last_product, 
            dtype=grad_output.dtype, 
            device=torch.cuda.current_device()
        ) if last_product else None
        grad_inner_product = torch.zeros_like(inner_product, dtype=grad_output.dtype, device=torch.cuda.current_device())
        grad_block_v = torch.zeros_like(sub_block_v[:, 1:-1], dtype=inner_product.dtype, device=torch.cuda.current_device())

        # calculate gradient for inner product
        grad_inner_product[:, :, :, 0:block_size] += torch.matmul(grad_output, sub_block_v[:, :-2].transpose(2, 3))
        grad_inner_product[:, :, :, block_size:(2 * block_size)] += torch.matmul(grad_output, sub_block_v[:, 1:-1].transpose(2, 3))
        grad_inner_product[:, :, :, (2 * block_size):(3 * block_size)] += torch.matmul(grad_output, sub_block_v[:, :2].transpose(2, 3))

        # compute gradients of first and last attention bands if applicable
        if grad_first_product is not None:
            grad_first_product[:, cur_start_block:(cur_end_block + 1)] += torch.matmul(grad_output[:, 0], sub_block_v[:, 1:-1].transpose(2, 3))
            grad_inner_product[:, :, :, (3 * block_size):(4 * block_size)] += torch.matmul(grad_output, sub_block_v[:, 1].transpose(2, 3))
        if grad_last_product is not None:
            grad_last_product[:, cur_start_block:(cur_end_block + 1)] += torch.matmul(grad_output[:, -1], sub_block_v[:, 1:-1].transpose(2, 3))
            grad_inner_product[:, :, :, (4 * block_size):(5 * block_size)] += torch.matmul(grad_output, sub_block_v[:, -1].transpose(2, 3))

        # calculate gradient of sub_block_v due to sliding window attention
        grad_block_v += torch.matmul(inner_product[:, :, :, block_size:(2 * block_size)].transpose(2, 3), grad_output)
        # if more than one block, part of left and right attention also used to compute output
        if local_blocks > 1:
            grad_block_v[:, :-1] += torch.matmul(inner_product[:, 1:, :, 0:block_size].transpose(2, 3), grad_output[:, 1:])
            grad_block_v[:, 1:] += torch.matmul(inner_product[:, :-1, :, (2 * block_size):(3 * block_size)].transpose(2, 3), grad_output[:, :-1])
        
        # use ring communication to calculate remaining parts of gradient
        first_a_block_grad_v = torch.matmul(inner_product[:, 0, :, 0:block_size].transpose(2, 3), grad_output[:, 0])
        last_a_block_grad_v = torch.matmul(inner_product[:, -1, :, (2 * block_size):(3 * block_size)].transpose(2, 3), grad_output[:, -1])
        for i in range(world_size - 1):
            sub_block_v = ring_forward(sub_block_v)
            start_block, end_block = _calc_incoming_device_block_range(i, local_rank, world_size)

            if grad_first_product is not None:
                grad_first_product[:, start_block:(end_block + 1)] += torch.matmul(grad_output[:, 0], sub_block_v[:, 1:-1].transpose(2, 3))
            if grad_last_product is not None:
                grad_last_product[:, start_block:(end_block + 1)] += torch.matmul(grad_output[:, -1], sub_block_v[:, 1:-1].transpose(2, 3))
            
            if start_block == 0:
                grad_inner_product[:, :, :, (3 * block_size):(4 * block_size)] += torch.matmul(grad_output, sub_block_v[:, 1].transpose(2, 3))
            if end_block == total_blocks - 1:
                grad_inner_product[:, :, :, (4 * block_size):(5 * block_size)] += torch.matmul(grad_output, sub_block_v[:, -1].transpose(2, 3))  

            if start_block == (cur_end_block + 1) % total_blocks:
                grad_block_v[:, -1] += first_a_block_grad_v
            if end_block == (cur_start_block - 1) % total_blocks:
                grad_block_v[:, 0] += last_a_block_grad_v
        
        # at this point, grad_block_v has the gradients from sliding window attention computed
        # computed gradients from global attention by first and last query blocks
        grad_block_v_from_global = torch.zeros(
            batch_size * num_heads,
            total_blocks,
            block_size,
            hidden_size,
            dtype=grad_block_v.dtype,
            device=torch.cuda.current_device()
        )
        if grad_first_product is not None:
            grad_block_v_from_global += torch.matmul(grad_first_product.transpose(2, 3), grad_output[:, 0])
        if grad_last_product is not None:
            grad_block_v_from_global += torch.matmul(grad_last_product.transpose(2, 3), grad_output[:, -1])
        torch.distributed.all_reduce(grad_block_v_from_global, group=get_tensor_model_parallel_group())
        grad_block_v_from_global = grad_block_v_from_global[:, cur_start_block:(cur_end_block + 1)]
        grad_block_v += grad_block_v_from_global

        # compute gradients from global attention by first and last query blocks
        grad_block_v_from_global = torch.matmul(
            inner_product[:, :, :, (3 * block_size):(4 * block_size)].transpose(2, 3), grad_output
        )
        grad_block_v_from_global = torch.sum(grad_block_v_from_global, dim=1)
        torch.distributed.reduce(grad_block_v_from_global, _calc_device_with_first_block(), group=get_tensor_model_parallel_group())
        if cur_start_block == 0:
            grad_block_v[:, 0] += grad_block_v_from_global
        grad_block_v_from_global = torch.matmul(
            inner_product[:, :, :, (4 * block_size):(5 * block_size)].transpose(2, 3), grad_output
        )
        grad_block_v_from_global = torch.sum(grad_block_v_from_global, dim=1)
        torch.distributed.reduce(grad_block_v_from_global, _calc_device_with_last_block(), group=get_tensor_model_parallel_group())
        if cur_end_block == total_blocks - 1:
            grad_block_v[:, -1] += grad_block_v_from_global

        # remove top and bottom attention gradients if global attention already accounts for them
        inner_block_range = _calc_current_device_inner_product_blocks(local_rank, total_blocks)
        if grad_first_product is not None and cur_start_block <= 1 <= cur_end_block:
            grad_inner_product[:, 0].fill_(0.0)
            grad_inner_product[:, 1, :, (3 * block_size):(4 * block_size)].fill_(0.0)
            grad_block_v[:, 0] /= world_size
            grad_block_v[:, 1] /= 4
        elif grad_first_product is not None:
            grad_inner_product[:, 0].fill_(0.0)
            grad_block_v[:, 0] /= world_size
        elif cur_start_block <= 1 <= cur_end_block:
            grad_inner_product[:, 0, :, (3 * block_size):(4 * block_size)].fill_(0.0)
            grad_block_v[:, 0] /= 4
        if grad_last_product is not None and cur_start_block <= total_blocks - 2 <= cur_end_block:
            grad_inner_product[:, -1].fill_(0.0)
            grad_inner_product[:, -2, :, (4 * block_size):(5 * block_size)].fill_(0.0)
            grad_block_v[:, -1] /= world_size
            grad_block_v[:, -2] /= 4
        elif last_product is not None:
            grad_inner_product[:, -1].fill_(0.0)
            grad_block_v[:, -1] /= world_size
        elif cur_start_block <= total_blocks - 2 <= cur_end_block:
            grad_inner_product[:, -1, :, (4 * block_size):(5 * block_size)].fill_(0.0)
            grad_block_v[:, -1] /= 4

        return grad_first_product, grad_inner_product, grad_last_product, grad_block_v

def check_bigbird_ring_qk(rank, world_size):
    global seq_length, sub_seq_length, block_size, batch_size, num_heads, hidden_size

    # create master tensors
    q = torch.rand(batch_size*num_heads, seq_length // block_size, block_size, hidden_size, dtype=torch.float16).cuda()
    k = torch.rand(batch_size*num_heads, seq_length // block_size, block_size, hidden_size, dtype=torch.float16).cuda()
    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)

    # create distributed tensors
    rank = dist.get_rank()
    start_block, end_block = _calc_current_device_block_range(rank)
    sub_q = q.clone()[:, start_block:(end_block + 1)].contiguous()
    sub_k = k.clone()[:, start_block:(end_block + 1)].contiguous()

    # set autograd attributes
    q.requires_grad = True
    k.requires_grad = True
    q.retain_grad()
    k.retain_grad()
    sub_q.requires_grad = True
    sub_k.requires_grad = True
    sub_q.retain_grad()
    sub_k.retain_grad()

    # compute master attention scores
    k = k.transpose(2, 3)
    master_first_product = torch.matmul(q[:, 0:1], k)
    master_last_product = torch.matmul(q[:, -1:], k)
    master_inner_product = torch.cat((
        torch.matmul(q, torch.cat((torch.roll(k, 1, 1), k, torch.roll(k, -1, 1)), dim=3)),
        torch.matmul(q, k[:, 0:1]),
        torch.matmul(q, k[:, -1:])),
        dim=3
    )
    master_inner_product[:, 0].fill_(-10000.0)
    master_inner_product[:, -1].fill_(-10000.0)
    master_inner_product[:, 1, :, (3 * block_size):(4 * block_size)].fill_(-10000.0)
    master_inner_product[:, -2, :, (4 * block_size):(5 * block_size)].fill_(-10000.0)

    # compute distributed attention scores
    ring_qk = BigBirdRingQK.apply
    (sub_first_product, sub_inner_product, sub_last_product) = ring_qk(sub_q, sub_k)
    sub_master_inner_product = master_inner_product[:, start_block:(end_block + 1)]
    
    # check master and distributed attention scores
    if sub_first_product is not None:
        assert torch.allclose(sub_first_product, master_first_product, rtol=1e-5, atol=1e-2), \
            'first product score does not match'
    if sub_last_product is not None:
        assert torch.allclose(sub_last_product, master_last_product, rtol=1e-5, atol=1e-2), \
            'last product score does not match'
    assert torch.allclose(sub_inner_product, sub_master_inner_product, rtol=1e-5, atol=1e-2), \
        'attention score does not match'

    # # run master backward
    # a.retain_grad()
    # a.mean().backward()

    # # run distributed backward
    # partial_master_a_grad = a.grad[:, rank*sub_seq_length:(rank+1)*sub_seq_length]
    # torch.autograd.backward(sub_a, partial_master_a_grad)

    # # check master and distributed grads
    # partial_master_q_grad = q.grad[:, rank*sub_seq_length:(rank+1)*sub_seq_length]
    # assert torch.allclose(sub_q.grad, partial_master_q_grad, rtol=1e-5, atol=1e-2), \
    #     'attention score cannot match'

def check_bigbird_ring_av(rank, world_size):
    global seq_length, sub_seq_length, block_size, batch_size, num_heads, hidden_size

    # create master tensors
    first_product = torch.rand(batch_size*num_heads, seq_length // block_size, block_size, block_size, dtype=torch.float16).cuda()
    inner_product = torch.rand(batch_size*num_heads, seq_length // block_size, block_size, 5 * block_size, dtype=torch.float16).cuda()
    inner_product[:, 0].fill_(0.0)
    inner_product[:, -1].fill_(0.0)
    inner_product[:, 1, :, (3 * block_size):(4 * block_size)].fill_(0.0)
    inner_product[:, -2, :, (4 * block_size):(5 * block_size)].fill_(0.0)
    last_product = torch.rand(batch_size*num_heads, seq_length // block_size, block_size, block_size, dtype=torch.float16).cuda()
    v = torch.rand(batch_size*num_heads, seq_length // block_size, block_size, hidden_size, dtype=torch.float16).cuda()
    dist.broadcast(first_product, src=0)
    dist.broadcast(inner_product, src=0)
    dist.broadcast(last_product, src=0)
    dist.broadcast(v, src=0)

    # create distributed tensors
    rank = dist.get_rank()
    start_block, end_block = _calc_current_device_block_range(rank)
    sub_first_product = first_product if start_block <= 0 <= end_block else None
    sub_inner_product = inner_product.clone()[:, start_block:(end_block + 1)].contiguous()
    sub_last_product = last_product if start_block <= (seq_length // block_size) - 1 <= end_block else None
    sub_v = v.clone()[:, start_block:(end_block + 1)].contiguous()

    # set autograd attributes
    first_product.requires_grad = True
    inner_product.requires_grad = True
    last_product.requires_grad = True
    v.requires_grad = True
    first_product.retain_grad()
    inner_product.retain_grad()
    last_product.retain_grad()
    v.retain_grad()
    if sub_first_product is not None:
        sub_first_product.requires_grad = True
        sub_first_product.retain_grad()
    if sub_last_product is not None:
        sub_last_product.requires_grad = True
        sub_last_product.retain_grad()
    sub_inner_product.requires_grad = True
    sub_inner_product.retain_grad()
    sub_v.requires_grad = True
    sub_v.retain_grad()

    # compute master output scores
    middle_out = torch.matmul(
        inner_product[:, 1:-1, :, :(3 * block_size)], 
        torch.cat((v[:, 0:-2], v[:, 1:-1], v[:, 2:]), dim=2)
    )
    middle_out += torch.matmul(inner_product[:, 1:-1, :, (3 * block_size):(4 * block_size)], v[:, 0:1])
    middle_out += torch.matmul(inner_product[:, 1:-1, :, (4 * block_size):(5 * block_size)], v[:, -1:])
    out = torch.cat(
        (
            torch.sum(torch.matmul(first_product, v), dim=1, keepdims=True),
            middle_out,
            torch.sum(torch.matmul(last_product, v), dim=1, keepdims=True)
        ),
        dim=1
    )

    # compute distributed attention scores
    ring_av = BigBirdRingAV.apply
    sub_out = ring_av(sub_first_product, sub_inner_product, sub_last_product, sub_v)
    sub_master_out = out[:, start_block:(end_block + 1)]
    
    # check master and distributed output scores
    assert torch.allclose(sub_out, sub_master_out, rtol=1e-5, atol=1e-2), \
        f'output score does not match {torch.eq(sub_out, sub_master_out)}'
    
    # # run master backward
    # a.retain_grad()
    # a.mean().backward()

    # # run distributed backward
    # partial_master_a_grad = a.grad[:, rank*sub_seq_length:(rank+1)*sub_seq_length]
    # torch.autograd.backward(sub_a, partial_master_a_grad)

    # # check master and distributed grads
    # partial_master_q_grad = q.grad[:, rank*sub_seq_length:(rank+1)*sub_seq_length]
    # assert torch.allclose(sub_q.grad, partial_master_q_grad, rtol=1e-5, atol=1e-2), \
    #     'attention score cannot match'

def run_test(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29100'

    # init process group
    dist.init_process_group(
        world_size=world_size,
        rank=rank,
        backend='nccl',
    )
    # bind process to a single GPU
    torch.cuda.set_device(rank)
    print(f'Rank {rank}: Initialization finished...')
    
    # start ring-QK test
    print(f'Rank {rank}: Starting BigBird-Ring-QK test...')
    check_bigbird_ring_qk(rank, world_size)
    print(f'Rank {rank}: Finished BigBird-Ring-QK test...')

    dist.barrier()

    # start ring-AV test
    print(f'Rank {rank}: Starting BigBird-Ring-AV test...')
    check_bigbird_ring_av(rank, world_size)
    print(f'Rank {rank}: Finished BigBird-Ring-AV test...')

    torch.cuda.empty_cache()

def test_sequence():
    global world_size
    run_func = partial(run_test, world_size=world_size)
    mp.spawn(run_func, nprocs=world_size)

if __name__ == '__main__':
    test_sequence()
