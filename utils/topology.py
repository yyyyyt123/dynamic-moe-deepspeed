import torch
import numpy

from deepspeed import comm as dist

# NUMBER OF TOTAL GPUS (WORLD_SIZE)
_TOTAL_GPUS_NUMBER=0
# NUMBER OF GPUS PER NODE
_GPUS_PER_NODE=0

def _get_total_gpu_number():
    assert _TOTAL_GPUS_NUMBER > 0, "total gpu numbers is 0, not correctly set topology info"
    
    return _TOTAL_GPUS_NUMBER

def _set_total_gpu_number():
    global _TOTAL_GPUS_NUMBER
    _TOTAL_GPUS_NUMBER = dist.get_world_size()

def _get_gpu_per_node_number():
    assert _GPUS_PER_NODE > 0, "total gpu numbers is 0, not correctly set topology info"
    
    return _GPUS_PER_NODE

def _set_gpu_per_node_number(number):
    global _GPUS_PER_NODE
    
    _GPUS_PER_NODE = 1 # for experiment only
    # _GPUS_PER_NODE = number # for experiment only
    # _GPUS_PER_NODE = torch.cuda.device_count()
    