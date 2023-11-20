""" Initialize the distributed environment. """
# Source: https://pytorch.org/tutorials/intermediate/dist_tuto.html

import torch.distributed as dist

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn()