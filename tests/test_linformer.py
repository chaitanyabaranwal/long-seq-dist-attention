import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import copy

from functools import partial

# params
world_size = 4
batch_size = 4
num_heads = 4
seq_length = 64
attention_head_size = 64
sub_seq_length = seq_length // world_size
linformer_k = 32

class LinformerRingQK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sub_q, sub_k):
        global batch_size, num_heads, sub_seq_length, linformer_k
        # create local segment for attention score
        attention_score = torch.zeros(
            batch_size * num_heads,
            sub_seq_length,
            linformer_k,
            dtype=sub_q.dtype,
            device=torch.cuda.current_device()
        )

        # collect all the projected key matrices into one
        dist.all_reduce(sub_k)

        # save tensors for backward pass
        ctx.save_for_backward(sub_q, sub_k)

        # compute local QK^T
        attention_score = torch.matmul(sub_q, sub_k)
        
        return attention_score

    @staticmethod
    def backward(ctx, grad_output):
        # Get arguments
        sub_q, sub_k = ctx.saved_tensors

        # calculate gradient of sub_q
        # we can directly calculate without communication because
        # the saved tensor was already the sum of individual tensors
        grad_q = torch.matmul(grad_output, sub_k.transpose(1, 2))

        # calculate gradient of sub_k
        grad_k = torch.matmul(sub_q.transpose(2, 1), grad_output)
        dist.all_reduce(grad_k)

        return grad_q, grad_k

class LinformerRingAV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, attention_score, sub_v):
        global batch_size, num_heads, sub_seq_length, attention_head_size
        # create local segment for attention result
        sub_attention_result = torch.zeros(
            batch_size * num_heads, 
            sub_seq_length,
            attention_head_size // num_heads,
            dtype=attention_score.dtype,
            device=torch.cuda.current_device()
        )

        # collect all the projected value matrices into one
        dist.all_reduce(sub_v)

        # save tensors for backward pass
        ctx.save_for_backward(attention_score, sub_v)

        # compute local AV
        sub_attention_result = torch.matmul(attention_score, sub_v)

        return sub_attention_result
    
    @staticmethod
    def backward(ctx, grad_output):
        # Get arguments
        attention_score, sub_v = ctx.saved_tensors

        # calculate gradient of attention_score
        # we can directly calculate without communication because
        # the saved tensor was already the sum of individual tensors
        grad_attention_score = torch.matmul(grad_output, sub_v.transpose(1, 2))

        # calculate gradient of sub_k
        grad_v = torch.matmul(attention_score.transpose(2, 1), grad_output)
        dist.all_reduce(grad_v)

        return grad_attention_score, grad_v

def check_linformer_ring_qk(rank, world_size):
    # create master tensors
    q = torch.rand(batch_size*num_heads, seq_length, attention_head_size, dtype=torch.float64).cuda()
    dist.broadcast(q, src=0)
    k = torch.rand(batch_size*num_heads, seq_length, attention_head_size, dtype=torch.float64).cuda()
    dist.broadcast(k, src=0)
    param = torch.rand(linformer_k, seq_length, dtype=torch.float64).cuda()
    parameter = torch.nn.Parameter(param)
    dist.broadcast(param, src=0)

    # create distributed tensors
    sub_q = q.clone()[:, rank*sub_seq_length:(rank+1)*sub_seq_length].detach()
    sub_k = k.clone()[:, rank*sub_seq_length:(rank+1)*sub_seq_length].detach()
    sub_param = param.clone()[:, rank*sub_seq_length:(rank+1)*sub_seq_length].detach()
    sub_parameter = torch.nn.Parameter(sub_param)
    sub_proj_k = torch.nn.functional.linear(sub_k.transpose(2, 1), sub_parameter)
    sub_proj_k_master = torch.nn.functional.linear(k.transpose(2, 1), parameter)

    # set autograd attributes
    parameter.requires_grad = True
    parameter.retain_grad()
    sub_parameter.requires_grad = True
    sub_parameter.retain_grad()
    q.requires_grad = True
    k.requires_grad = True
    q.retain_grad()
    k.retain_grad()
    sub_q.requires_grad = True
    sub_k.requires_grad = True
    sub_q.retain_grad()
    sub_k.retain_grad()
    sub_proj_k.retain_grad()
    sub_proj_k_master.retain_grad()

    # compute distributed attention scores
    sub_a = LinformerRingQK.apply(sub_q, sub_proj_k)
    
    # compute master attention scores
    a = torch.matmul(q, sub_proj_k_master)

    # check master and distributed attention scores
    sub_master_a = a[:, rank*sub_seq_length:(rank+1)*sub_seq_length]
    assert torch.allclose(sub_a, sub_master_a, rtol=1e-5, atol=1e-8), \
        'attention score does not match'

    # run master backward
    a.retain_grad()
    a.mean().backward()

    # run distributed backward
    partial_master_a_grad = a.grad[:, rank*sub_seq_length:(rank+1)*sub_seq_length]
    torch.autograd.backward(sub_a, partial_master_a_grad)

    # check master and distributed grads
    partial_master_q_grad = q.grad[:, rank*sub_seq_length:(rank+1)*sub_seq_length]
    assert torch.allclose(sub_q.grad, partial_master_q_grad, rtol=1e-5), \
        'partial Q gradient does not match'

    partial_master_param_grad = parameter.grad[:, rank*sub_seq_length:(rank+1)*sub_seq_length]
    assert torch.allclose(partial_master_param_grad, sub_parameter.grad, rtol=1e-5), \
        'partial parameter gradient does not match'

    partial_master_k_grad = torch.matmul(sub_proj_k_master.grad, parameter)[:, :, rank*sub_seq_length:(rank+1)*sub_seq_length]
    sub_k_grad = torch.matmul(sub_proj_k.grad, sub_parameter)
    assert torch.allclose(sub_k_grad, partial_master_k_grad, rtol=1e-5), \
        'partial K gradient does not match'

def check_linformer_ring_av(rank, world_size):
    # create master tensors
    a = torch.rand(batch_size*num_heads, seq_length, linformer_k, dtype=torch.float64).cuda()
    dist.broadcast(a, src=0)
    v = torch.rand(batch_size*num_heads, seq_length, attention_head_size, dtype=torch.float64).cuda()
    dist.broadcast(v, src=0)
    param = torch.rand(linformer_k, seq_length, dtype=torch.float64).cuda()
    parameter = torch.nn.Parameter(param)
    dist.broadcast(param, src=0)

    # create distributed tensors
    sub_a = a.clone()[:, rank*sub_seq_length:(rank+1)*sub_seq_length].detach()
    sub_v = v.clone()[:, rank*sub_seq_length:(rank+1)*sub_seq_length].detach()
    sub_param = param.clone()[:, rank*sub_seq_length:(rank+1)*sub_seq_length].detach()
    sub_parameter = torch.nn.Parameter(sub_param)
    sub_proj_v = torch.nn.functional.linear(sub_v.transpose(2, 1), sub_parameter)
    sub_proj_v_master = torch.nn.functional.linear(v.transpose(2, 1), parameter)

    # set autograd attributes
    parameter.requires_grad = True
    parameter.retain_grad()
    sub_parameter.requires_grad = True
    sub_parameter.retain_grad()
    a.requires_grad = True
    v.requires_grad = True
    a.retain_grad()
    v.retain_grad()
    sub_a.requires_grad = True
    sub_v.requires_grad = True
    sub_a.retain_grad()
    sub_v.retain_grad()
    sub_proj_v.retain_grad()
    sub_proj_v_master.retain_grad()

    # compute distributed attention scores
    sub_out = LinformerRingAV.apply(sub_a, sub_proj_v.transpose(2, 1))

    # compute master attention scores
    out = torch.matmul(a, sub_proj_v_master.transpose(2, 1))

    # check master and distributed attention scores
    sub_master_out = out[:, rank*sub_seq_length:(rank+1)*sub_seq_length]
    assert torch.allclose(sub_out, sub_master_out, rtol=1e-5, atol=1e-8), \
        'output score does not match'

    # run master backward
    out.retain_grad()
    out.mean().backward()

    # run distributed backward
    partial_master_out_grad = out.grad[:, rank*sub_seq_length:(rank+1)*sub_seq_length]
    torch.autograd.backward(sub_out, partial_master_out_grad)

    # check master and distributed grads
    partial_master_a_grad = a.grad[:, rank*sub_seq_length:(rank+1)*sub_seq_length]
    assert torch.allclose(sub_a.grad, partial_master_a_grad, rtol=1e-5), \
        'partial A gradient does not match'

    partial_master_param_grad = parameter.grad[:, rank*sub_seq_length:(rank+1)*sub_seq_length]
    assert torch.allclose(partial_master_param_grad, sub_parameter.grad, rtol=1e-5), \
        'partial parameter gradient does not match'

    partial_master_v_grad = torch.matmul(sub_proj_v_master.grad, parameter)[:, :, rank*sub_seq_length:(rank+1)*sub_seq_length]
    sub_v_grad = torch.matmul(sub_proj_v.grad, sub_parameter)
    assert torch.allclose(sub_v_grad, partial_master_v_grad, rtol=1e-5), \
        'partial V gradient does not match'


def run_test(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    # how to initialize?
    dist.init_process_group(
        world_size=world_size,
        rank=rank,
        backend='nccl',
    )

    # bind process to a single GPU
    torch.cuda.set_device(rank)

    print(f'Rank {rank}: Initialization finished...')
    
    print(f'Rank {rank}: Starting Linformer-Ring-QK test...')
    check_linformer_ring_qk(rank, world_size)
    print(f'Rank {rank}: Finished Linformer-Ring-QK test...')
    print(f'Rank {rank}: Starting Linformer-Ring-AV test...')
    check_linformer_ring_av(rank, world_size)
    print(f'Rank {rank}: Finished Linformer-Ring-AV test...')

    torch.cuda.empty_cache()


def test_sequence():
    global world_size
    run_func = partial(run_test, world_size=world_size)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_sequence()
