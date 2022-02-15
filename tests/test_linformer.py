import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest

from functools import partial

def check_linformer_ring_qk(rank, world_size):
    # params
    batch_size = 4
    num_heads = 4
    seq_length = 64
    attention_head_size = 64
    sub_seq_length = seq_length // world_size
    linformer_k = 32

    # create master tensors
    q = torch.rand(batch_size*num_heads, seq_length, attention_head_size).cuda()
    dist.broadcast(q, src=0, group=???)
    sub_k = torch.rand(batch_size*num_heads, attention_head_size, linformer_k).cuda()
    k = sub_k.clone().detach()
    dist.all_reduce(k, group=???)

    # create distributed tensors
    sub_q = q.clone()[:, rank*sub_seq_length:(rank+1)*sub_seq_length].contiguous()

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
    a = torch.matmul(q, k)

    # compute distributed attention scores
    linformer_ring_qk = megatron.mpu.layers.LinformerRingQK.apply
    sub_a = linformer_ring_qk(sub_q, sub_k)

    # check master and distributed attention scores
    sub_master_a = a[:, rank*sub_seq_length:(rank+1)*sub_seq_length]
    assert torch.allclose(sub_a, sub_master_a, rtol=1e-5, atol=1e-2), \
        'attention score does not match'

    # run master backward
    a.retain_grad()
    a.mean().backward()

    # run distributed backward
    partial_master_a_grad = a.grad[:, rank*sub_seq_length:(rank+1)*sub_seq_length]
    torch.autograd.backward(sub_a, partial_master_a_grad)

    # check master and distributed grads
    partial_master_q_grad = q.grad[:, rank*sub_seq_length:(rank+1)*sub_seq_length]
    assert torch.allclose(sub_q.grad, partial_master_q_grad, rtol=1e-5, atol=1e-2), \
        'partial Q gradient does not match'
    sub_k_grad = sub_k.grad.clone()
    dist.all_reduce(sub_k_grad, group=???)
    assert torch.allclose(sub_k_grad, k.grad, rol=1e-5, ato=1e-2), \
        'sum of partial K gradients does not match master K gradient'


def check_linformer_ring_av(rank, world_size):
    # params
    batch_size = 4
    num_heads = 4
    seq_length = 64
    attention_head_size = 64
    sub_seq_length = seq_length // world_size
    linformer_k = 32

    # create master tensors
    a = torch.rand(batch_size*num_heads, seq_length, linformer_k).cuda()
    dist.broadcast(q, src=0, group=???)
    sub_v = torch.rand(batch_size*num_heads, linformer_k, attention_head_size).cuda()
    v = sub_k.clone().detach()
    dist.all_reduce(v, group=???)

    # create distributed tensors
    sub_a = a.clone()[:, rank*sub_seq_length:(rank+1)*sub_seq_length].contiguous()

    # set autograd attributes
    a.requires_grad = True
    v.requires_grad = True
    a.retain_grad()
    v.retain_grad()
    sub_a.requires_grad = True
    sub_v.requires_grad = True
    sub_a.retain_grad()
    sub_v.retain_grad()

    # compute master attention scores
    out = torch.matmul(a, v)

    # compute distributed attention scores
    linformer_ring_av = megatron.mpu.layers.LinformerRingAV.apply
    sub_out = linformer_ring_av(sub_a, sub_v)

    # check master and distributed attention scores
    sub_master_out = out[:, rank*sub_seq_length:(rank+1)*sub_seq_length]
    assert torch.allclose(sub_out, sub_master_out, rtol=1e-5, atol=1e-2), \
        'output score does not match'

    # run master backward
    out.retain_grad()
    out.mean().backward()

    # run distributed backward
    partial_master_out_grad = out.grad[:, rank*sub_seq_length:(rank+1)*sub_seq_length]
    torch.autograd.backward(sub_out, partial_master_out_grad)

    # check master and distributed grads
    partial_master_a_grad = a.grad[:, rank*sub_seq_length:(rank+1)*sub_seq_length]
    assert torch.allclose(sub_a.grad, partial_master_a_grad, rtol=1e-5, atol=1e-2), \
        'partial A gradient does not match'
    sub_v_grad = sub_v.grad.clone()
    dist.all_reduce(sub_v_grad, group=???)
    assert torch.allclose(sub_v_grad, v.grad, rol=1e-5, ato=1e-2), \
        'sum of partial V gradients does not match master V gradient'


def run_test(rank, world_size):
    # how to initialize?

    
    check_linformer_ring_qk(rank, world_size)
    check_linformer_ring_av(rank, world_size)

    torch.cuda.empty_cache()


@pytest.mark.dist
def test_sequence():
    world_size = 4
    run_func = partial(run_test, world_size=world_size)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_sequence()
