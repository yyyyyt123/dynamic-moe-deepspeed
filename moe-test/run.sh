#!/bin/bash

# Number of nodes
NUM_NODES=1
# Number of GPUs per node
NUM_GPUS=2
# Size of expert parallel world (should be less than total world size)
EP_SIZE=2
# Number of total experts
EXPERTS=2

# dynamic gating top_k 0
deepspeed --num_nodes=${NUM_NODES} --num_gpus=${NUM_GPUS} dynamic_moe.py > log.txt


