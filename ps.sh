
#!/bin/bash

# Set the number of threads to 1
export OMP_NUM_THREADS=1

# Set the Python virtual environment
source ./venv/bin/activate

# Run torchrun with the provided number of processes
torchrun --nproc-per-node $1 ps.py