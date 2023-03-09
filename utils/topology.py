import torch
import numpy

from deepspeed import comm as dist

# NUMBER OF TOTAL GPUS (WORLD_SIZE)
_TOTAL_GPUS_NUMBER=0
# NUMBER OF GPUS PER NODE
_GPUS_PER_NODE=0
# (for all_reduce) experts in dynamic expert parallel groups / global tokens
_DYNAMIC_CURRENT_EXPERT_PROPORTION={}

""" topology related """
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
    # for experiment only
    _GPUS_PER_NODE = number 
    
    # for real-world cases
    # _GPUS_PER_NODE = torch.cuda.device_count()

""" proportion of current local experts tokens"""    
def _init_dynamic_experts_tokens_proportion(current_experts_name):
    
    global _DYNAMIC_CURRENT_EXPERT_PROPORTION
    for n in current_experts_name:
        _DYNAMIC_CURRENT_EXPERT_PROPORTION[n] = 0.0
    
def _get_dynamic_experts_tokens_proportion(expert_name):
    """Get proportion of processed tokens of current expert"""
    res = _DYNAMIC_CURRENT_EXPERT_PROPORTION[expert_name]
    # res should be less than 1.0 (for all_reduce)
    assert res <= 1.0, f"current proportation of experts processed tokens:{res} are not normalized!"
    return res

def _count_dynamic_experts_tokens_proportion(expert_name, tokens_num):
    """Set proportion of processed tokens of current expert"""
    global _DYNAMIC_CURRENT_EXPERT_PROPORTION
    _DYNAMIC_CURRENT_EXPERT_PROPORTION[expert_name] += tokens_num
    
def _normalize_dynamic_experts_tokens_proportion(expert_name, total_tokens):
    global _DYNAMIC_CURRENT_EXPERT_PROPORTION
    t = _DYNAMIC_CURRENT_EXPERT_PROPORTION[expert_name]
    _DYNAMIC_CURRENT_EXPERT_PROPORTION[expert_name] = t / total_tokens
    
    # for debug
    # print(f"rank:{dist.get_rank()}, proportion for {expert_name} is {t / total_tokens}")
    
def _reset_dynamic_experts_tokens_proportion():
    global _DYNAMIC_CURRENT_EXPERT_PROPORTION
    for k,v in _DYNAMIC_CURRENT_EXPERT_PROPORTION.items():
        _DYNAMIC_CURRENT_EXPERT_PROPORTION[k]=0
    
