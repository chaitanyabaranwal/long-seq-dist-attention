import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import random

from functools import partial

# params
world_size = 4
batch_size = 4
num_heads = 4
seq_length = 128
hidden_size = 64
sub_seq_length = seq_length // world_size
block_size = 8

def ring_backward(tensor_send_prev):
    global world_size

    ops = []

    current_rank = dist.get_rank()
    # print(f'current rank: {current_rank}, next rank: {mpu.get_tensor_model_parallel_next_rank()}, prev rank: {mpu.get_tensor_model_parallel_prev_rank()}')

    tensor_recv_next = torch.empty_like(tensor_send_prev,
                                   requires_grad=True,
                                   device=torch.cuda.current_device())

    # send to prev rank
    send_prev_op = dist.P2POp(
        dist.isend, tensor_send_prev,
        (current_rank - 1) % world_size)
    ops.append(send_prev_op)

    # receive from next rank
    recv_next_op = dist.P2POp(
        dist.irecv, tensor_recv_next,
        (current_rank + 1) % world_size)
    ops.append(recv_next_op)

    if current_rank % 2 == 0:
        ops = ops[::-1]

    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()

    # To protect against race condition when using batch_isend_irecv().
    torch.cuda.synchronize()

    return tensor_recv_next

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
    def forward(ctx, sub_block_q, sub_block_k, random_mapping):
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
            7 * block_size,
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
        if world_size > 1:
            first_q_left_block = torch.empty(
                batch_size * num_heads,
                1,
                hidden_size,
                block_size,
                dtype=sub_block_q.dtype,
                device=torch.cuda.current_device()
            )
            last_q_right_block = torch.empty_like(first_q_left_block)
        else:
            first_q_left_block = sub_block_k[:, -1:].transpose(2, 3)
            last_q_right_block = sub_block_k[:, 0:1].transpose(2, 3)


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
        
        # check random attention
        for i in range(local_blocks):
            if cur_start_block <= random_mapping[i][0] <= cur_end_block:
                inner_product[:, i, :, (5 * block_size):(6 * block_size)] = torch.matmul(
                    sub_block_q[:, i], sub_block_k[:, random_mapping[i][0] - cur_start_block]
                )
            if cur_start_block <= random_mapping[i][1] <= cur_end_block:
                inner_product[:, i, :, (6 * block_size):(7 * block_size)] = torch.matmul(
                    sub_block_q[:, i], sub_block_k[:, random_mapping[i][1] - cur_start_block]
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
            
            # check random attention
            for j in range(local_blocks):
                if start_block <= random_mapping[j][0] <= end_block:
                    inner_product[:, j, :, (5 * block_size):(6 * block_size)] = torch.matmul(
                        sub_block_q[:, j], sub_block_k[:, random_mapping[j][0] - start_block]
                    )
                if start_block <= random_mapping[j][1] <= end_block:
                    inner_product[:, j, :, (6 * block_size):(7 * block_size)] = torch.matmul(
                        sub_block_q[:, j], sub_block_k[:, random_mapping[j][1] - start_block]
                    )

            # gather any remaining blocks needed for sliding window attention
            if start_block == (cur_end_block + 1) % total_blocks:
                last_q_right_block = sub_block_k[:, 0:1]
            if end_block == (cur_start_block - 1) % total_blocks:
                first_q_left_block = sub_block_k[:, -1:]
        
        # get back original block
        if world_size > 1:
            sub_block_k = ring_forward(sub_block_k)
        # concatenate any extra key blocks needed for sliding window attention
        sub_block_k = torch.cat((first_q_left_block, sub_block_k, last_q_right_block), dim=1)

        # save tensor for backward
        ctx.save_for_backward(sub_block_q, sub_block_k, random_mapping)

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
        return first_product, inner_product, last_product
    
    @staticmethod
    def backward(ctx, grad_first_product, grad_inner_product, grad_last_product):
        # get saved tensors
        sub_block_q, sub_block_k, random_mapping = ctx.saved_tensors
        sub_block_k = sub_block_k.transpose(2, 3)

        # Get arguments
        global world_size, batch_size, seq_length, sub_seq_length, hidden_size, block_size, num_heads
        local_rank = dist.get_rank()
        total_blocks = seq_length // block_size
        local_blocks = sub_seq_length // block_size
        cur_start_block, cur_end_block = _calc_current_device_block_range(local_rank)

        # setup tensors for the gradients
        grad_block_q = torch.zeros_like(sub_block_q, dtype=grad_inner_product.dtype)
        grad_block_k = torch.zeros(
            batch_size * num_heads,
            total_blocks,
            block_size,
            hidden_size,
            dtype=grad_inner_product.dtype,
            device=torch.cuda.current_device()
        )

        # calculate local gradients of sub_block_q based on sliding window attention
        # TODO (chai): Consider breaking down into parts to save memory
        grad_block_q += torch.matmul(
            grad_inner_product[:, :, :, :(3 * block_size)],
            torch.cat((sub_block_k[:, :-2], sub_block_k[:, 1:-1], sub_block_k[:, 2:]), dim=2)
        )

        # calculate local gradients of sub_block_q based on global attention
        if grad_first_product is not None:
            grad_block_q[:, 0] += torch.sum(torch.matmul(grad_first_product[:, cur_start_block:(cur_end_block + 1)], sub_block_k[:, 1:-1]), dim=1)
            grad_block_q += torch.matmul(grad_inner_product[:, :, :, (3 * block_size):(4 * block_size)], sub_block_k[:, 1:2])
        if grad_last_product is not None:
            grad_block_q[:, -1] += torch.sum(torch.matmul(grad_last_product[:, cur_start_block:(cur_end_block + 1)], sub_block_k[:, 1:-1]), dim=1)
            grad_block_q += torch.matmul(grad_inner_product[:, :, :, (4 * block_size):(5 * block_size)], sub_block_k[:, -2:-1])
        
        # calculate local gradients of sub_block_q based on random attention
        for i in range(local_blocks):
            if cur_start_block <= random_mapping[i][0] <= cur_end_block:
                grad_block_q[:, i] += torch.matmul(
                    grad_inner_product[:, i, :, (5 * block_size):(6 * block_size)],
                    sub_block_k[:, random_mapping[i][0] - cur_start_block + 1]
                )
            if cur_start_block <= random_mapping[i][1] <= cur_end_block:
                grad_block_q[:, i] += torch.matmul(
                    grad_inner_product[:, i, :, (6 * block_size):(7 * block_size)],
                    sub_block_k[:, random_mapping[i][1] - cur_start_block + 1]
                )
        
        # compute the grad_block_q and grad_block_k blocks using ring communication
        for i in range(world_size - 1):
            sub_block_k = ring_forward(sub_block_k)
            start_block, end_block = _calc_incoming_device_block_range(i, local_rank, world_size)

            # first and last blocks pay attention to all blocks
            if grad_first_product is not None:
                grad_block_q[:, 0] += torch.sum(torch.matmul(grad_first_product[:, start_block:(end_block + 1)], sub_block_k[:, 1:-1]), dim=1)
            if grad_last_product is not None:
                grad_block_q[:, -1] += torch.sum(torch.matmul(grad_last_product[:, start_block:(end_block + 1)], sub_block_k[:, 1:-1]), dim=1)

            # first and last block gets attention from all blocks
            if start_block == 0:
                grad_block_q += torch.matmul(grad_inner_product[:, :, :, (3 * block_size):(4 * block_size)], sub_block_k[:, 1:2])
            if end_block == total_blocks - 1:
                grad_block_q += torch.matmul(grad_inner_product[:, :, :, (4 * block_size):(5 * block_size)], sub_block_k[:, -2:-1])

            # calculate local gradients of sub_block_q based on random attention
            for j in range(local_blocks):
                if start_block <= random_mapping[j][0] <= end_block:
                    grad_block_q[:, j] += torch.matmul(
                        grad_inner_product[:, j, :, (5 * block_size):(6 * block_size)],
                        sub_block_k[:, random_mapping[j][0] - start_block + 1]
                    )
                if start_block <= random_mapping[j][1] <= end_block:
                    grad_block_q[:, j] += torch.matmul(
                        grad_inner_product[:, j, :, (6 * block_size):(7 * block_size)],
                        sub_block_k[:, random_mapping[j][1] - start_block + 1]
                    )
        
        # At this point, grad_block_q has been computed
        # calculate gradient of sub_block_k
        
        # second sliding window attention band
        grad_block_k[:, cur_start_block:(cur_end_block + 1)] += torch.matmul(
            grad_inner_product[:, :, :, block_size:(2 * block_size)].transpose(2, 3), 
            sub_block_q
        )
        # first sliding window attention band
        if grad_first_product is not None:
            grad_block_k[:, cur_start_block:cur_end_block] += torch.matmul(
                grad_inner_product[:, 1:, :, 0:block_size].transpose(2, 3), sub_block_q[:, 1:]
            )
            grad_block_k[:, -1] += torch.matmul(
                grad_inner_product[:, 0, :, 0:block_size].transpose(1, 2), sub_block_q[:, 0]
            )
        else:
            grad_block_k[:, (cur_start_block - 1):cur_end_block] += torch.matmul(
                grad_inner_product[:, :, :, 0:block_size].transpose(2, 3), sub_block_q
            )
        # third sliding window attention band
        if grad_last_product is not None:
            grad_block_k[:, (cur_start_block + 1):(cur_end_block + 1)] += torch.matmul(
                grad_inner_product[:, :-1, :, (2 * block_size):(3 * block_size)].transpose(2, 3), sub_block_q[:, :-1]
            )
            grad_block_k[:, 0] += torch.matmul(
                grad_inner_product[:, -1, :, (2 * block_size):(3 * block_size)].transpose(1, 2), sub_block_q[:, -1]
            )
        else:
            grad_block_k[:, (cur_start_block + 1):(cur_end_block + 2)] += torch.matmul(
                grad_inner_product[:, :, :, (2 * block_size):(3 * block_size)].transpose(2, 3), sub_block_q
            )
        
        # compute gradients from random attention
        for i in range(local_blocks):
            grad_block_k[:, random_mapping[i][0]] += torch.matmul(
                grad_inner_product[:, i, :, (5 * block_size):(6 * block_size)].transpose(1, 2),
                sub_block_q[:, i]
            )
            grad_block_k[:, random_mapping[i][1]] += torch.matmul(
                grad_inner_product[:, i, :, (6 * block_size):(7 * block_size)].transpose(1, 2),
                sub_block_q[:, i]
            )

        # compute gradients from global attention which attends to every key block
        if grad_first_product is not None:
            grad_block_k += torch.matmul(grad_first_product.transpose(2, 3), sub_block_q[:, 0:1])
        if grad_last_product is not None:
            grad_block_k += torch.matmul(grad_last_product.transpose(2, 3), sub_block_q[:, -1:])

        # compute gradients from global attention which is attends to first and last key blocks
        grad_block_k[:, 0] += torch.sum(
            torch.matmul(grad_inner_product[:, :, :, (3 * block_size):(4 * block_size)].transpose(2, 3), sub_block_q),
            dim=1
        )
        grad_block_k[:, -1] += torch.sum(
            torch.matmul(grad_inner_product[:, :, :, (4 * block_size):(5 * block_size)].transpose(2, 3), sub_block_q),
            dim=1
        )
        dist.all_reduce(grad_block_k)
        grad_block_k = grad_block_k[:, cur_start_block:(cur_end_block + 1)]

        return grad_block_q, grad_block_k, None

class BigBirdRingAV(torch.autograd.Function):
    """
    Calculates the sparse AV in a ring-exchange style.
    The resultant attention matrix is a collection of blocks of output values.
    """

    @staticmethod
    def forward(ctx, first_product, inner_product, last_product, sub_block_v, random_mapping):
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
        if world_size > 1:
            first_a_left_block = torch.empty(
                batch_size * num_heads,
                1,
                block_size,
                hidden_size,
                dtype=sub_block_v.dtype,
                device=torch.cuda.current_device()
            )
            last_a_right_block = torch.empty_like(first_a_left_block)
        else:
            first_a_left_block = sub_block_v[:, -1:]
            last_a_right_block = sub_block_v[:, 0:1]

        # first and last attention block attend to all value blocks
        if first_context is not None:
            first_context += torch.sum(torch.matmul(first_product[:, cur_start_block:(cur_end_block + 1)], sub_block_v), dim=1, keepdims=True)
            inner_context += torch.matmul(inner_product[:, :, :, (3 * block_size):(4 * block_size)], sub_block_v[:, 0:1])
        if last_context is not None:
            last_context += torch.sum(torch.matmul(last_product[:, cur_start_block:(cur_end_block + 1)], sub_block_v), dim=1, keepdims=True)
            inner_context += torch.matmul(inner_product[:, :, :, (4 * block_size):(5 * block_size)], sub_block_v[:, -1:])

        # check random attention
        for i in range(local_blocks):
            if cur_start_block <= random_mapping[i][0] <= cur_end_block:
                inner_context[:, i] += torch.matmul(
                    inner_product[:, i, :, (5 * block_size):(6 * block_size)], 
                    sub_block_v[:, random_mapping[i][0] - cur_start_block]
                )
            if cur_start_block <= random_mapping[i][1] <= cur_end_block:
                inner_context[:, i] += torch.matmul(
                    inner_product[:, i, :, (6 * block_size):(7 * block_size)], 
                    sub_block_v[:, random_mapping[i][1] - cur_start_block]
                )
        
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

            # check random attention
            for j in range(local_blocks):
                if start_block <= random_mapping[j][0] <= end_block:
                    inner_context[:, j] += torch.matmul(
                        inner_product[:, j, :, (5 * block_size):(6 * block_size)], 
                        sub_block_v[:, random_mapping[j][0] - start_block]
                    )
                if start_block <= random_mapping[j][1] <= end_block:
                    inner_context[:, j] += torch.matmul(
                        inner_product[:, j, :, (6 * block_size):(7 * block_size)], 
                        sub_block_v[:, random_mapping[j][1] - start_block]
                    )
            
            # gather any remaining value blocks needed for sliding window attention
            if start_block == (cur_end_block + 1) % total_blocks:
                last_a_right_block = sub_block_v[:, 0:1]
            if end_block == (cur_start_block - 1) % total_blocks:
                first_a_left_block = sub_block_v[:, -1:]
        
        if world_size > 1:
            # get back original block
            sub_block_v = ring_forward(sub_block_v)
        # concatenate any extra value blocks for sliding window attention
        sub_block_v = torch.cat((first_a_left_block, sub_block_v, last_a_right_block), dim=1)

        # save tensor for backward
        ctx.save_for_backward(first_product, inner_product, last_product, sub_block_v, random_mapping)

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
        first_product, inner_product, last_product, sub_block_v, random_mapping = ctx.saved_tensors

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
        ) if first_product is not None else None
        grad_last_product = torch.zeros_like(
            last_product, 
            dtype=grad_output.dtype, 
            device=torch.cuda.current_device()
        ) if last_product is not None else None
        grad_inner_product = torch.zeros_like(inner_product, dtype=grad_output.dtype, device=torch.cuda.current_device())
        grad_block_v = torch.zeros(
            batch_size * num_heads,
            total_blocks,
            block_size,
            hidden_size,
            dtype=inner_product.dtype,
            device=torch.cuda.current_device()
        )

        # calculate gradient for inner product
        # TODO (chai): Consider breaking down into parts to save memory
        grad_inner_product[:, :, :, :(3 * block_size)] += torch.matmul(
            grad_output,
            torch.cat((sub_block_v[:, :-2], sub_block_v[:, 1:-1], sub_block_v[:, 2:]), dim=2).transpose(2, 3)
        )

        # compute gradients of first and last attention bands if applicable
        if grad_first_product is not None:
            grad_first_product[:, cur_start_block:(cur_end_block + 1)] += torch.matmul(grad_output[:, 0:1], sub_block_v[:, 1:-1].transpose(2, 3))
            grad_inner_product[:, :, :, (3 * block_size):(4 * block_size)] += torch.matmul(grad_output, sub_block_v[:, 1:2].transpose(2, 3))
        if grad_last_product is not None:
            grad_last_product[:, cur_start_block:(cur_end_block + 1)] += torch.matmul(grad_output[:, -1:], sub_block_v[:, 1:-1].transpose(2, 3))
            grad_inner_product[:, :, :, (4 * block_size):(5 * block_size)] += torch.matmul(grad_output, sub_block_v[:, -2:-1].transpose(2, 3))

        # calculate gradients of inner_product based on random attention
        for i in range(local_blocks):
            if cur_start_block <= random_mapping[i][0] <= cur_end_block:
                grad_inner_product[:, i, :, (5 * block_size):(6 * block_size)] += torch.matmul(
                    grad_output[:, i],
                    sub_block_v[:, random_mapping[i][0] - cur_start_block + 1].transpose(1, 2)
                )
            if cur_start_block <= random_mapping[i][1] <= cur_end_block:
                grad_inner_product[:, i, :, (6 * block_size):(7 * block_size)] += torch.matmul(
                    grad_output[:, i],
                    sub_block_v[:, random_mapping[i][1] - cur_start_block + 1].transpose(1, 2)
                )
        
        # use ring communication to calculate remaining parts of gradient
        for i in range(world_size - 1):
            sub_block_v = ring_forward(sub_block_v)
            start_block, end_block = _calc_incoming_device_block_range(i, local_rank, world_size)

            if grad_first_product is not None:
                grad_first_product[:, start_block:(end_block + 1)] += torch.matmul(grad_output[:, 0:1], sub_block_v[:, 1:-1].transpose(2, 3))
            if grad_last_product is not None:
                grad_last_product[:, start_block:(end_block + 1)] += torch.matmul(grad_output[:, -1:], sub_block_v[:, 1:-1].transpose(2, 3))
            
            if start_block == 0:
                grad_inner_product[:, :, :, (3 * block_size):(4 * block_size)] += torch.matmul(grad_output, sub_block_v[:, 1:2].transpose(2, 3))
            if end_block == total_blocks - 1:
                grad_inner_product[:, :, :, (4 * block_size):(5 * block_size)] += torch.matmul(grad_output, sub_block_v[:, -2:-1].transpose(2, 3))  

            # calculate gradients of inner_product based on random attention
            for j in range(local_blocks):
                if start_block <= random_mapping[j][0] <= end_block:
                    grad_inner_product[:, j, :, (5 * block_size):(6 * block_size)] += torch.matmul(
                        grad_output[:, j],
                        sub_block_v[:, random_mapping[j][0] - start_block + 1].transpose(1, 2)
                    )
                if start_block <= random_mapping[j][1] <= end_block:
                    grad_inner_product[:, j, :, (6 * block_size):(7 * block_size)] += torch.matmul(
                        grad_output[:, j],
                        sub_block_v[:, random_mapping[j][1] - start_block + 1].transpose(1, 2)
                    )

        # at this point, gradient of attention products has been computed
        # calculate gradient of sub_block_v

        # second band of sliding window attention
        grad_block_v[:, cur_start_block:(cur_end_block + 1)] += torch.matmul(
            inner_product[:, :, :, block_size:(2 * block_size)].transpose(2, 3), 
            grad_output
        )
        # first band of sliding window attention
        if first_product is not None:
            grad_block_v[:, cur_start_block:cur_end_block] += torch.matmul(
                inner_product[:, 1:, :, 0:block_size].transpose(2, 3), 
                grad_output[:, 1:]
            )
            grad_block_v[:, -1] += torch.matmul(
                inner_product[:, 0, :, 0:block_size].transpose(1, 2), 
                grad_output[:, 0]
            )
        else:
            grad_block_v[:, (cur_start_block - 1):cur_end_block] += torch.matmul(
                inner_product[:, :, :, 0:block_size].transpose(2, 3), 
                grad_output
            )
        # last band of sliding window attention
        if last_product is not None:
            grad_block_v[:, (cur_start_block + 1):(cur_end_block + 1)] += torch.matmul(
                inner_product[:, :-1, :, (2 * block_size):(3 * block_size)].transpose(2, 3), 
                grad_output[:, :-1]
            )
            grad_block_v[:, 0] += torch.matmul(
                inner_product[:, -1, :, (2 * block_size):(3 * block_size)].transpose(1, 2), 
                grad_output[:, -1]
            )
        else:
            grad_block_v[:, (cur_start_block + 1):(cur_end_block + 2)] += torch.matmul(
                inner_product[:, :, :, (2 * block_size):(3 * block_size)].transpose(2, 3), 
                grad_output
            )
        
        # compute gradients from random attention
        for i in range(local_blocks):
            grad_block_v[:, random_mapping[i][0]] += torch.matmul(
                inner_product[:, i, :, (5 * block_size):(6 * block_size)].transpose(1, 2),
                grad_output[:, i]
            )
            grad_block_v[:, random_mapping[i][1]] += torch.matmul(
                inner_product[:, i, :, (6 * block_size):(7 * block_size)].transpose(1, 2),
                grad_output[:, i]
            )

        # compute gradients from global attention attending to every key block
        if grad_first_product is not None:
            grad_block_v += torch.matmul(first_product.transpose(2, 3), grad_output[:, 0:1])
        if grad_last_product is not None:
            grad_block_v += torch.matmul(last_product.transpose(2, 3), grad_output[:, -1:])

        # compute gradients from global attention attending to first and last key blocks
        grad_block_v[:, 0] += torch.sum(
            torch.matmul(inner_product[:, :, :, (3 * block_size):(4 * block_size)].transpose(2, 3), grad_output),
            dim=1
        )
        grad_block_v[:, -1] += torch.sum(
            torch.matmul(inner_product[:, :, :, (4 * block_size):(5 * block_size)].transpose(2, 3), grad_output),
            dim=1
        )
        dist.all_reduce(grad_block_v)
        grad_block_v = grad_block_v[:, cur_start_block:(cur_end_block + 1)]

        # remove top and bottom attention gradients if global attention already accounts for them
        inner_block_range = _calc_current_device_inner_product_blocks(local_rank, total_blocks)
        if grad_first_product is not None and cur_start_block <= 1 <= cur_end_block:
            grad_inner_product[:, 0].fill_(0.0)
            grad_inner_product[:, 1, :, (3 * block_size):(4 * block_size)].fill_(0.0)
        elif grad_first_product is not None:
            grad_inner_product[:, 0].fill_(0.0)
        elif cur_start_block <= 1 <= cur_end_block:
            grad_inner_product[:, 0, :, (3 * block_size):(4 * block_size)].fill_(0.0)
        if grad_last_product is not None and cur_start_block <= total_blocks - 2 <= cur_end_block:
            grad_inner_product[:, -1].fill_(0.0)
            grad_inner_product[:, -2, :, (4 * block_size):(5 * block_size)].fill_(0.0)
        elif last_product is not None:
            grad_inner_product[:, -1].fill_(0.0)
        elif cur_start_block <= total_blocks - 2 <= cur_end_block:
            grad_inner_product[:, -1, :, (4 * block_size):(5 * block_size)].fill_(0.0)

        return grad_first_product, grad_inner_product, grad_last_product, grad_block_v, None

def get_bigbird_random_block_mapping(total_blocks, num_random_blocks):
    '''
    Gets a tensor mapping each block to its two random blocks, and the blocks which use current block.
    '''
    # Create list values
    final_map = []
    for i in range(total_blocks):
        if i == 0:
            invalid_vals = [0, 1, total_blocks - 1]
        elif i == 1:
            invalid_vals = [0, 1, 2, total_blocks - 1]
        elif i == total_blocks - 2:
            invalid_vals = [total_blocks - 3, total_blocks - 2, total_blocks - 1, 0]
        elif i == total_blocks - 1:
            invalid_vals = [total_blocks - 2, total_blocks - 1, 0]
        else:
            invalid_vals = [0, i - 1, i, i + 1, total_blocks - 1]
        valid_vals = list(set(range(total_blocks)) - set(invalid_vals))
        blocks = random.sample(valid_vals, num_random_blocks)
        final_map.append(blocks)

    # Copy to tensor
    return torch.cuda.LongTensor(final_map)

def check_bigbird_ring_qk(rank, world_size):
    global seq_length, sub_seq_length, block_size, batch_size, num_heads, hidden_size

    # create master tensors
    q = torch.rand(batch_size*num_heads, seq_length // block_size, block_size, hidden_size, dtype=torch.float64).cuda()
    k = torch.rand(batch_size*num_heads, seq_length // block_size, block_size, hidden_size, dtype=torch.float64).cuda()
    random_mapping = get_bigbird_random_block_mapping(seq_length // block_size, 2)
    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(random_mapping, src=0)

    # create distributed tensors
    rank = dist.get_rank()
    start_block, end_block = _calc_current_device_block_range(rank)
    sub_q = q.clone()[:, start_block:(end_block + 1)].contiguous()
    sub_k = k.clone()[:, start_block:(end_block + 1)].contiguous()
    sub_random_mapping = random_mapping[start_block:(end_block + 1), :]

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
    master_first_product = torch.matmul(q[:, 0:1], k.transpose(2, 3))
    master_last_product = torch.matmul(q[:, -1:], k.transpose(2, 3))
    master_random_product = torch.empty(batch_size*num_heads, seq_length // block_size, block_size, 2 * block_size, dtype=torch.float64).cuda()
    for i in range(seq_length // block_size):
        master_random_product[:, i, :, :block_size] = torch.matmul(q[:, i], k[:, random_mapping[i][0]].transpose(1, 2))
        master_random_product[:, i, :, block_size:(2 * block_size)] = torch.matmul(q[:, i], k[:, random_mapping[i][1]].transpose(1, 2))
    master_inner_product = torch.cat((
        torch.matmul(q, torch.cat((torch.roll(k.transpose(2, 3), 1, 1), k.transpose(2, 3), torch.roll(k.transpose(2, 3), -1, 1)), dim=3)),
        torch.matmul(q, k.transpose(2, 3)[:, 0:1]),
        torch.matmul(q, k.transpose(2, 3)[:, -1:]),
        master_random_product),
        dim=3
    )
    master_inner_product[:, 0].fill_(-10000.0)
    master_inner_product[:, -1].fill_(-10000.0)
    master_inner_product[:, 1, :, (3 * block_size):(4 * block_size)].fill_(-10000.0)
    master_inner_product[:, -2, :, (4 * block_size):(5 * block_size)].fill_(-10000.0)

    # compute distributed attention scores
    ring_qk = BigBirdRingQK.apply
    (sub_first_product, sub_inner_product, sub_last_product) = ring_qk(sub_q, sub_k, sub_random_mapping)
    sub_master_inner_product = master_inner_product[:, start_block:(end_block + 1)]
    
    # check master and distributed attention scores
    if sub_first_product is not None:
        assert torch.allclose(sub_first_product, master_first_product), \
            'first product score does not match'
    if sub_last_product is not None:
        assert torch.allclose(sub_last_product, master_last_product), \
            'last product score does not match'
    assert torch.allclose(sub_inner_product, sub_master_inner_product), \
        'inner product score does not match'

    # run master backward
    master_first_product.retain_grad()
    master_inner_product.retain_grad()
    master_last_product.retain_grad()
    master_first_product.mean().backward()
    master_inner_product.mean().backward()
    master_last_product.mean().backward()
    master_inner_product.grad[:, 0].fill_(0.0)
    master_inner_product.grad[:, 1, :, (3 * block_size):(4 * block_size)].fill_(0.0)
    master_inner_product.grad[:, -1].fill_(0.0)
    master_inner_product.grad[:, -2, :, (4 * block_size):(5 * block_size)].fill_(0.0)

    # run distributed backward
    partial_master_first_product_grad = master_first_product.grad if sub_first_product is not None else None
    partial_master_inner_product_grad = master_inner_product.grad[:, start_block:(end_block + 1)]
    partial_master_last_product_grad = master_last_product.grad if sub_last_product is not None else None
    if sub_first_product is not None and sub_last_product is not None:
        torch.autograd.backward(
            (sub_first_product, sub_inner_product, sub_last_product), 
            (partial_master_first_product_grad, partial_master_inner_product_grad, partial_master_last_product_grad)
        )
    elif sub_first_product is not None:
        torch.autograd.backward(
            (sub_first_product, sub_inner_product), 
            (partial_master_first_product_grad, partial_master_inner_product_grad)
        )
    elif sub_last_product is not None:
        torch.autograd.backward(
            (sub_inner_product, sub_last_product), 
            (partial_master_inner_product_grad, partial_master_last_product_grad)
        )
    else:
        torch.autograd.backward(sub_inner_product, partial_master_inner_product_grad)

    # check master and distributed grads
    partial_master_q_grad = q.grad[:, start_block:(end_block + 1)]
    assert torch.allclose(sub_q.grad, partial_master_q_grad), \
        'partial Q gradient does not match'
    partial_master_k_grad = k.grad[:, start_block:(end_block + 1)]
    assert torch.allclose(sub_k.grad, partial_master_k_grad), \
        'partial K gradient does not match'

def check_bigbird_ring_av(rank, world_size):
    global seq_length, sub_seq_length, block_size, batch_size, num_heads, hidden_size

    # create master tensors
    first_product = torch.rand(batch_size*num_heads, seq_length // block_size, block_size, block_size, dtype=torch.float64).cuda()
    inner_product = torch.rand(batch_size*num_heads, seq_length // block_size, block_size, 7 * block_size, dtype=torch.float64).cuda()
    inner_product[:, 0].fill_(0.0)
    inner_product[:, -1].fill_(0.0)
    inner_product[:, 1, :, (3 * block_size):(4 * block_size)].fill_(0.0)
    inner_product[:, -2, :, (4 * block_size):(5 * block_size)].fill_(0.0)
    last_product = torch.rand(batch_size*num_heads, seq_length // block_size, block_size, block_size, dtype=torch.float64).cuda()
    v = torch.rand(batch_size*num_heads, seq_length // block_size, block_size, hidden_size, dtype=torch.float64).cuda()
    random_mapping = get_bigbird_random_block_mapping(seq_length // block_size, 2)
    dist.broadcast(first_product, src=0)
    dist.broadcast(inner_product, src=0)
    dist.broadcast(last_product, src=0)
    dist.broadcast(v, src=0)
    dist.broadcast(random_mapping, src=0)

    # create distributed tensors
    rank = dist.get_rank()
    start_block, end_block = _calc_current_device_block_range(rank)
    sub_first_product = first_product if start_block <= 0 <= end_block else None
    sub_inner_product = inner_product.clone()[:, start_block:(end_block + 1)].contiguous()
    sub_last_product = last_product if start_block <= (seq_length // block_size) - 1 <= end_block else None
    sub_v = v.clone()[:, start_block:(end_block + 1)].contiguous()
    sub_random_mapping = random_mapping[start_block:(end_block + 1), :]

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
    for i in range(1, seq_length // block_size - 1):
        middle_out[:, i - 1] += torch.matmul(inner_product[:, i, :, (5 * block_size):(6 * block_size)], v[:, random_mapping[i][0]])
        middle_out[:, i - 1] += torch.matmul(inner_product[:, i, :, (6 * block_size):(7 * block_size)], v[:, random_mapping[i][1]])
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
    sub_out = ring_av(sub_first_product, sub_inner_product, sub_last_product, sub_v, sub_random_mapping)
    sub_master_out = out[:, start_block:(end_block + 1)]
    
    # check master and distributed output scores
    assert torch.allclose(sub_out, sub_master_out), \
        'output score does not match'
    
    # # run master backward
    out.retain_grad()
    out.mean().backward()

    # run distributed backward
    partial_master_out_grad = out.grad[:, start_block:(end_block + 1)]
    torch.autograd.backward(sub_out, partial_master_out_grad)
    inner_product.grad[:, 0].fill_(0.0)
    inner_product.grad[:, -1].fill_(0.0)
    inner_product.grad[:, 1, :, (3 * block_size):(4 * block_size)].fill_(0.0)
    inner_product.grad[:, -2, :, (4 * block_size):(5 * block_size)].fill_(0.0)

    # check master and distributed grads
    if sub_first_product is not None:
        assert torch.allclose(sub_first_product.grad, first_product.grad), \
            'first product gradient does not match'
    if sub_last_product is not None:
        assert torch.allclose(sub_last_product.grad, last_product.grad), \
            'last product gradient does not match'
    partial_master_inner_product_grad = inner_product.grad[:, start_block:(end_block + 1)]
    assert torch.allclose(sub_inner_product.grad, partial_master_inner_product_grad), \
        'inner product gradient does not match'
    
    partial_master_v_grad = v.grad[:, start_block:(end_block + 1)]
    assert torch.allclose(sub_v.grad, partial_master_v_grad), \
        'partial V gradient does not match'

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
