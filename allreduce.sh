
#!/bin/bash

# Set the number of threads to 1
export OMP_NUM_THREADS=1

# If you're on mac, uncomment next line and replace ./venv/bin/activate by : venv/bin/activate
# python3 -m venv venv
# Set the Python virtual environment
source ./venv/bin/activate
# Install requirements
pip3 install -r requirements.txt

# Run torchrun with the provided number of processes
torchrun --nproc-per-node $1 allreduce.py
