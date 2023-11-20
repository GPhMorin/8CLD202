""" Peer-to-Peer SGD Example """
# By: Gilles-Philippe Morin
# Based on [1], [2], [3], and [4]
# Usage: torchrun --nproc-per-node {number of processes} p2p.py
# export OMP_NUM_THREADS=1 to limit to one thread per process

# Sources:
# [1] https://pytorch.org/tutorials/intermediate/dist_tuto.html
# [2] https://github.com/pytorch/examples/blob/main/mnist/main.py
# [3] https://gist.github.com/Praneet9
# [4] https://github.com/LPD-EPFL/Garfield/tree/master/pytorch_impl/applications/LEARN

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.multiprocessing import Process
from torchvision import datasets, transforms

import time
from net import Net
from test import test
from init_process import init_process
import torch.distributed.rpc as rpc
from torch.utils.data.distributed import DistributedSampler


def get_parameters():
    """ Return the model parameters. """
    global model
    return [param for param in model.parameters()]


def average_parameters(model):
    """ Parameter averaging. """
    global peers
    future_peers = peers
    for i in peers:
        try:
            params = rpc.rpc_sync(f"worker{i}", get_parameters, timeout=1)
            for local_params, remote_params in zip(model.parameters(), params):
                buffer = (local_params.data + remote_params.data) / 2
                local_params.data.copy_(buffer)
        except RuntimeError:
            future_peers.remove(i)
    peers = future_peers


def train(optimizer, model, train_set, epoch):
    rank = dist.get_rank()
    epoch_loss = 0.0
    number_of_images = 0
    for data, target in train_set:
        number_of_images += len(data)
    for batch_idx, (data, target) in enumerate(train_set):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        epoch_loss += loss.data.item()
        loss.backward()
#        if rank == 0:
#            print("Before: ", [param for param in model.parameters()][0])
        average_parameters(model)
#        if rank == 0:
#            print("After: ", [param for param in model.parameters()][0])
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train: Rank {} Epoch {} [{}/{} ({:.0f}%)] Loss {:.6f}'.format(
                rank, epoch, batch_idx * len(data), number_of_images,
                100. * batch_idx / len(train_set), loss.item()))


def run_worker():
    """ Run the Worker """
    global model
    global peers
    size = dist.get_world_size()
    rank = dist.get_rank()
    rpc.init_rpc(f"worker{rank}", rank=rank, world_size=size)
    peers = list(range(size))
    peers.remove(rank)
    train_set = datasets.MNIST('../data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    sampler = DistributedSampler(train_set)
    train_loader = torch.utils.data.DataLoader(train_set,
        batch_size=32, sampler=sampler)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                           ])),
                           batch_size=32, shuffle=True)
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    for epoch in range(3):
        sampler.set_epoch(epoch)
        train(optimizer, model, train_loader, epoch)
        if rank == 0:
            test(model, test_loader)
    rpc.shutdown()


def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn()


if __name__ == "__main__":
    start = time.time()

    address = os.environ['MASTER_ADDR']
    port = os.environ['MASTER_PORT']
    rank = int(os.environ['RANK'])
    size = int(os.environ['WORLD_SIZE'])
    
    processes = []
    p = Process(target=init_process, args=(rank, size, run_worker))
    p.start()
    processes.append(p)
    for p in processes:
        p.join()

    if rank == 0:
        print(f"Time taken: {time.time() - start :.2f} seconds")