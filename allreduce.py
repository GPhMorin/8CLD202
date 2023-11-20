""" AllReduce Distributed Synchronous SGD Example """
# By: Gilles-Philippe Morin
# Based on [1], [2], and [3]
# Usage: torchrun --nproc-per-node {number of processes} allreduce.py
# export OMP_NUM_THREADS=1 to limit to one thread per process

# Sources:
# [1] https://pytorch.org/tutorials/intermediate/dist_tuto.html
# [2] https://github.com/pytorch/examples/blob/main/mnist/main.py
# [3] https://gist.github.com/Praneet9

import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim

from torch.multiprocessing import Process
from torchvision import datasets, transforms

from time import time
from net import Net
from test import test
from init_process import init_process
from torch.utils.data.distributed import DistributedSampler


def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
#    rank = dist.get_rank()
    for param in model.parameters():
#        if rank == 0:
#            print("Before: ", param)
        dist.all_reduce(param)
#        if rank == 0:
#            print("After: ", param)
        param.data.div_(size)
#        if rank == 0:
#            print("After2: ", param)


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
        average_gradients(model)
        optimizer.step()
        if batch_idx % 10 == 0 and rank == 0:
            print('Train: Rank {} Epoch {} [{}/{} ({:.0f}%)] Loss {:.6f}'.format(
                rank, epoch, batch_idx * len(data), number_of_images,
                100. * batch_idx / len(train_set), loss.item()))


def run_worker():
    """ Distributed Synchronous SGD Example """
    rank = dist.get_rank()
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


if __name__ == "__main__":
    start = time()

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
        print(f"Time taken: {time() - start :.2f} seconds")