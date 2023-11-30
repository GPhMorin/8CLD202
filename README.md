# Comparison of AllReduce, Parameter Server and Peer-to-Peer Architectures for Distributed Machine Learning

By Gilles-Philippe MORIN (MORG27109707), Kévin SILLIAU (SILK30070300), and Lucca MASI (MASL05080300) for the course 8CLD202 "Infonuagique" at Université du Québec à Chicoutimi.

This project aims to refactor a single-machine, single-process machine learning model into a single-machine, multi-process model using distributed data parallelization.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

This setup was tested on Ubuntu 23.10.

First you need to set up a virtual Python environment in the local folder `venv`. For this project, we used Python 3.11.3.

You will need to `pip install torch torchvision` in your virtual environment. For this project, we used PyTorch 2.1.1.

The bash scripts run automatically the venv and install the requirements, so make sure to make them exetutable (chmod u+x script.sh) to run them (see Usage part below).

We added Mac OS Support (for Apple Silicon Chips) : make sure to check the .sh scripts, uncomment the necessary lines and follow the instructions to make them functionnal on Mac.

## Usage

In a terminal, `cd` inside the folder and run the appropriate bash script with the number of processes as argument. For instance:

- If you want to run the AllReduce implementation with three workers, run `./allreduce.sh 3`.

- If you want to run the Parameter Server implementation with one server and two clients, run `./ps.sh 3`.

- If you want to run the Peer-to-Peer implementation with three pairs of servers and clients, run `./p2p.sh 3`.

The first time one of the three scripts is run, you will require an Internet connection to download the MNIST dataset in the `data` folder.
