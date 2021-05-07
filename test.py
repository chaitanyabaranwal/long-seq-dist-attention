import torch
import os

from pyroute2 import NDB

def ring_forward(tensor_send_next):
    buffer_shape = tensor_send_next.size()
    dtype = tensor_send_next.dtype

    ops = []

    current_rank = torch.distributed.get_rank()
    print(f'current rank: {current_rank}, next rank: {(current_rank+1)%4}, prev rank: {(current_rank-1)%4}', flush=True)

    tensor_recv_prev = torch.empty(buffer_shape,
                                   requires_grad=True,
                                   device=torch.cuda.current_device(),
                                   dtype=dtype)

    # send to next rank
    send_next_op = torch.distributed.P2POp(
        torch.distributed.isend, tensor_send_next,
        (current_rank + 1) % 4)
    ops.append(send_next_op)

    # receive from prev rank
    recv_prev_op = torch.distributed.P2POp(
        torch.distributed.irecv, tensor_recv_prev,
        (current_rank - 1) % 4)
    ops.append(recv_prev_op)

    if current_rank % 2 == 0:
        ops = ops[::-1]

    reqs = torch.distributed.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()

    print('here')
    # To protect against race condition when using batch_isend_irecv().
    torch.cuda.synchronize()

    return tensor_recv_prev


if __name__ == '__main__':
    print('start', flush=True)
    ndb = NDB(log='on')

    for record in ndb.addresses.summary():
        record_dict = record._as_dict()

        if record_dict['ifname'] == 'ib0' and ':' not in record_dict['address']:
            address = record_dict['address']
            break

    print(f'found IP address: {address}', flush=True)
    os.environ['MASTER_ADDR'] = address
    os.environ['MASTER_PORT'] = '29500'

    rank = int(os.environ['PMI_RANK'])
    world_size = int(os.environ['PMI_SIZE'])

    print(f'rank: {rank}, world_size: {world_size}', flush=True)

    torch.distributed.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size
        )
    print(f'initialized on rank {rank}', flush=True)
    torch.cuda.set_device(rank)

    data = torch.rand(4, 16).cuda(torch.cuda.current_device())
    ring_forward(data)