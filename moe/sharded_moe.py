'''
Copyright 2021 The Microsoft DeepSpeed Team
'''
# The file has been adapted from two fairscale files:
# (1) https://github.com/facebookresearch/fairscale/blob/master/fairscale/nn/moe/moe_layer.py
# (2) https://github.com/facebookresearch/fairscale/blob/master/fairscale/nn/moe/top2gate.py
# Git commit hash: 34df606902a240567a0d898037ece55c2f1336cf
# We retain the following license from the original files:

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from deepspeed.utils.timer import SynchronizedWallClockTimer
from deepspeed.utils import logger, topology
from typing import Callable, Dict, TYPE_CHECKING, Any, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from deepspeed.utils import groups
from .mappings import drop_tokens, gather_tokens
from deepspeed.ops.op_builder import UtilsBuilder

util_ops = UtilsBuilder().load()
flatten = util_ops.flatten
unflatten = util_ops.unflatten

if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module

uniform_map: Dict[torch.device, Callable] = {}
gumbel_map: Dict[torch.device, Callable] = {}
exp_selection_uniform_map: Dict[torch.device, Callable] = {}

try:
    # To enable Tutel MoE optimizations:
    #   python3 -m pip install --user --upgrade git+https://github.com/microsoft/tutel@v0.1.x
    from tutel import moe as tutel_moe
    TUTEL_INSTALLED = True
except:
    # Fail silently so we don't spam logs unnecessarily if user isn't using tutel
    TUTEL_INSTALLED = False
    pass


def multiplicative_jitter(x, device: torch.device, epsilon=1e-2):
    """
    Modified from switch transformer paper. mesh transformers
    Multiply values by a random number between 1-epsilon and 1+epsilon.
    Makes models more resilient to rounding errors introduced by bfloat16.
    This seems particularly important for logits.
    Args:
        x: a torch.tensor
        device: torch.device
        epsilon: a floating point value
    Returns:
        a jittered x.
    """
    if epsilon == 0:
        return x
    uniform = uniform_map.get(device)
    if uniform is None:
        uniform = torch.distributions.uniform.Uniform(
            low=torch.tensor(1.0 - epsilon,
                             device=device),
            high=torch.tensor(1.0 + epsilon,
                              device=device)).rsample  # type: ignore
        uniform_map[device] = uniform
    return x * uniform(x.shape)


def gumbel_rsample(shape: Tuple, device: torch.device) -> Tensor:
    gumbel = gumbel_map.get(device)
    if gumbel is None:
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample  # type: ignore
        gumbel_map[device] = gumbel
    return gumbel(shape)


from deepspeed import comm as dist

# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.


# Based on https://github.com/pytorch/pytorch/pull/40762
class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx: Any,
            # TODO: replace with DS process group
            group: torch.distributed.ProcessGroup,
            input: Tensor) -> Tensor:  # type: ignore
        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        dist.all_to_all_single(output, input, group=group)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, _AllToAll.apply(ctx.group, *grad_output))

class _AllToAll_UNEQUAL(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx: Any,
            group: torch.distributed.ProcessGroup,
            input: Tensor,
            input_split: list,
            output_split: list) -> Tensor:  # type: ignore
        ctx.group = group
        ctx.input_split=input_split
        ctx.output_split=output_split

        # input = input.contiguous()
        output = torch.zeros(sum(output_split), device=input.device, dtype=input.dtype)
        
        dist.all_to_all_single(output=output, tensor=input, output_split_sizes=output_split, 
                               input_split_sizes=input_split, group=group)
        
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        # print(f"rank:{dist.get_rank()}:before all2all, group:{ctx.group}")
        result = _AllToAll_UNEQUAL.apply(ctx.group, grad_output[0], 
                                              ctx.output_split, ctx.input_split)
        # print(f"rank:{dist.get_rank()}:after all2all, group:{ctx.group}")

        return (None,
                result,
                None,
                None)
    

# einsum rewrites are on par or more performant
# switch can be bubbled up in future
USE_EINSUM = True


# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.
def einsum(rule, a, b):
    if USE_EINSUM:
        return torch.einsum(rule, a, b)
    elif rule == 's,se->se':
        return a.reshape(a.shape[0], -1) * b
    elif rule == 'se,sc->sec':
        return a.unsqueeze(2) * b.unsqueeze(1)
    elif rule == 'se,se->s':
        return torch.bmm(a.unsqueeze(1), b.unsqueeze(2)).reshape(-1)
    elif rule == 'sec,sm->ecm':
        s = a.shape[0]
        e = a.shape[1]
        c = a.shape[2]
        m = b.shape[1]
        return torch.matmul(a.reshape(s, -1).t(), b).reshape(e, c, m)
    elif rule == 'sec,ecm->sm':
        return torch.matmul(a.reshape(a.shape[0], -1), b.reshape(-1, b.shape[-1]))
    elif rule == 'ks,ksm->sm':
        k = b.shape[0]
        s = b.shape[1]
        m = b.shape[2]
        # [k, s] -> [s, k] -> [s, 1, k]
        a = a.t().unsqueeze(1)
        # [k,s,m] -> [k, sm] -> [sm, k] -> [s, m, k]
        b = b.reshape(k, -1).t().reshape(s, m, k)
        # bmm([s, 1, k], [s, m, k]^t) -> [s, m, 1]
        return torch.bmm(a, b.transpose(1, 2)).squeeze(2)
    else:
        return torch.einsum(rule, a, b)


# The following functions are extracted and scripted
# because otherwise during a torch.jit.trace, the non-Tensor
# values used in the calculations get recorded as constants.
# torch.jit.script coerces them into Tensors and preserves
# their dynamic shapes. This enables ONNX export.
# We can't script the entire top1gating function because it
# includes stateful caching logic which is incompatible with ONNX.


@torch.jit.script
def _capacity(gates: Tensor, capacity_factor: Tensor, min_capacity: Tensor) -> Tensor:
    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    # to(torch.int64) works around a bug in torch.onnx.export:
    # it should cast k to int64 when converting torch.topk but it doesn't.
    capacity = torch.ceil((num_tokens / num_experts) * capacity_factor).to(torch.int64)
    if capacity < min_capacity:
        capacity = min_capacity.to(torch.int64)
    return capacity


@torch.jit.script
def _top_idx(source, k):
    return torch.topk(source, k=k, dim=0)[1]  # 按列找topk大，返回的是n行k列，内容为topk的具体值出现在原本序列中的位置(最后的[1])


@torch.jit.script
def _one_hot_to_float(x, num_classes):
    return F.one_hot(x, num_classes=num_classes).float()


def top1gating(logits: Tensor, # tokens after embedding
               capacity_factor: float,
               min_capacity: int,
               used_token: Tensor = None,
               noisy_gate_policy: Optional[str] = None,
               drop_tokens: bool = True,
               use_rts: bool = True,
               use_tutel: bool = False) -> Tuple[Tensor,
                                                 Tensor,
                                                 Tensor,
                                                 Tensor]:
    """Implements Top1Gating on logits."""
    if noisy_gate_policy == 'RSample':
        logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    # everything is in fp32 in this function
    gates = F.softmax(logits, dim=1) # gates.shape (num of tokens, num of experts) 按照experts的得分做softmax

    capacity = _capacity(gates,
                         torch.tensor(capacity_factor),
                         torch.tensor(min_capacity))
    # logger.warning("capacity:{}".format(capacity))

    # Create a mask for 1st's expert per token
    # noisy gating
    indices1_s = torch.argmax(  # 按行，选择top1
        logits_w_noise if noisy_gate_policy == 'RSample' else gates,
        dim=1)
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)
    
    # logger.warning("mask1.shape({})".format(mask1.shape)) # 8, 2
    # logger.warning("mask1({})".format(mask1)) # 8, 2
    
    # mask only used tokens
    if used_token is not None:
        mask1 = einsum("s,se->se", used_token, mask1) # 按列相乘，不求和 s:num of tokens e: num of experts 

    # gating decisions
    exp_counts = torch.sum(mask1, dim=0).detach().to('cpu') # shape:(1,e) 保存了发给每个experts的tokens数量,不受capacity限制

    # logger.warning("exp_counts.shape({})".format(exp_counts.shape))
    # logger.warning("exp_counts({})".format(exp_counts))

    # if we don't want to drop any tokens
    if not drop_tokens:
        new_capacity = torch.max(exp_counts).to(logits.device)
        dist.all_reduce(new_capacity, op=dist.ReduceOp.MAX, group=dist.get_world_group())  # all reduce以group=world，求world中所有的new_capacity的最大值
        capacity = new_capacity

    # Compute l_aux
    me = torch.mean(gates, dim=0) # 按列求均值 -> 所有tokens，在每个expert的权重的平均分 
    ce = torch.mean(mask1.float(), dim=0) # 每个experts上，平均通过的tokens数量
    l_aux = torch.sum(me * ce) * num_experts

    # Random Token Selection
    if use_rts:
        uniform = exp_selection_uniform_map.get(logits.device)
        if uniform is None:
            uniform = torch.distributions.uniform.Uniform( # 均匀分布
                low=torch.tensor(0.0,
                                 device=logits.device),
                high=torch.tensor(1.0,
                                  device=logits.device)).rsample
            exp_selection_uniform_map[logits.device] = uniform

        mask1_rand = mask1 * uniform(mask1.shape)  # 均匀分布
    else:
        mask1_rand = mask1

    # logger.warning("mask1_rand.shape({})".format(mask1_rand.shape)) # 8,2
    # logger.warning("mask1_rand({})".format(mask1_rand)) # 8,2


    assert logits.shape[0] >= min_capacity, "No. of tokens (batch-size) should be greater than min_capacity. Either set min_capacity to 0 or increase your batch size."

    top_idx = _top_idx(mask1_rand, capacity) # 返回 capacity行，expert列，数值为：发送给每个expert的具体tokens的位置(mask1_rand中，按列从大到小排序的下标)

    # logger.warning("top_idx.shape({})".format(top_idx.shape)) # 4, 2
    # logger.warning(f"top_idx({top_idx})") # 4, 2

    new_mask1 = mask1 * torch.zeros_like(mask1).scatter_(0, top_idx, 1) # dim=0，表示在top_idx元素行号用其元素值代替，列号不变，以这两个值索引，在结果new_mask1中填1
    mask1 = new_mask1

    # logger.warning("new_mask1.shape({})".format(new_mask1.shape))
    # logger.warning("new_mask1({})".format(new_mask1))

    if use_tutel:
        # Tutel doesn't support index values masked with zero
        # so we need to replace masked indices with -1
        indices_mask = mask1.sum(dim=1) * num_experts - 1
        indices1_s = torch.min(indices1_s, indices_mask)

    # Compute locations in capacity buffer
    # location1 shape:(num of tokens, num of experts) 表示了每个token的dispatch对于experts capacity的变化
    if use_tutel:
        locations1 = tutel_moe.fast_cumsum_sub_one(mask1)
    else:
        locations1 = torch.cumsum(mask1, dim=0) - 1

    # logger.warning("locations1.shape({})".format(locations1.shape))
    # logger.warning("locations1({})".format(locations1))

    if use_tutel:
        gates1_s = (gates * mask1).sum(dim=1)
        locations1_s = torch.sum(locations1 * mask1, dim=1)
        return l_aux, capacity, num_experts, [indices1_s,], [locations1_s,], [gates1_s,], exp_counts

    # Store the capacity location for each token
    # 求和是为了记录每个tokens经过dispatch后，对于expert capacity的大小
    locations1_s = torch.sum(locations1 * mask1, dim=1)  # 经过 *mask 操作， 每行只有一个非零元素（代表第k列的expert的capacity情况）

    # logger.warning("locations1_s.shape({})".format(locations1_s.shape))
    # logger.warning("locations1_s({})".format(locations1_s))
    
    # Normalize gate probabilities
    mask1_float = mask1.float()
    gates = gates * mask1_float  # gates代表具体dispatch到哪个expert，第k列代表dispatch给第k个experts的tokens
    
    # logger.warning("gates({})".format(gates))

    locations1_sc = _one_hot_to_float(locations1_s, capacity) # location_sc代表每个tokens在experts capacity的one_hot编码, 第k列代表占用的capacity是第几个
    # logger.warning("locations1_sc.shape({})".format(locations1_sc.shape))
    # logger.warning("locations1_sc({})".format(locations1_sc))
    # einsum之后得到mask的表
    combine_weights = einsum("se,sc->sec", gates, locations1_sc)
    
    # logger.warning("combine_weights.shape({})".format(combine_weights.shape))
    # logger.warning("combine_weights({})".format(combine_weights))
    

    dispatch_mask = combine_weights.bool()
    
    # logger.warning("dispatch_mask.shape({})".format(dispatch_mask.shape))
    # logger.warning("dispatch_mask({})".format(dispatch_mask))
    
    # exit(0)
    

    return l_aux, combine_weights, dispatch_mask, exp_counts


def top2gating(logits: Tensor,
               capacity_factor: float,
               min_capacity: int) -> Tuple[Tensor,
                                           Tensor,
                                           Tensor,
                                           Tensor]:
    """Implements Top2Gating on logits."""
    # everything is in fp32 in this function
    gates = F.softmax(logits, dim=1)

    capacity = _capacity(gates,
                         torch.tensor(capacity_factor * 2),
                         torch.tensor(min_capacity))

    # logger.debug("gates.shape:{}".format(gates.shape))
    # logger.debug("gates:{}".format(gates))
    # logger.debug("capacity:{}".format(capacity))
    # logger.debug("capacity_factor:{}".format(capacity_factor*2))
    
    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1)
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)
    # logger.debug("mask1.shape({})".format(mask1.shape)) # 8，4
    # logger.debug("mask1({})".format(mask1)) # 

    # Create a mask for 2nd's expert per token using Gumbel-max trick
    # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
    logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    # Replace top-expert with min value
    logits_except1 = logits_w_noise.masked_fill(mask1.bool(), float("-inf"))
    indices2_s = torch.argmax(logits_except1, dim=1)
    mask2 = F.one_hot(indices2_s, num_classes=num_experts)

    # logger.debug("indices2_s({})".format(indices2_s))  
    # logger.debug("mask2({})".format(mask2))  
    # logger.debug("logits_except1.shape({})".format(logits_except1.shape))  
    # logger.debug("logits({})".format(logits))  
    # logger.debug("gumbel_rsample(logits.shape)({})".format(gumbel_rsample(logits.shape, device=logits.device)))  
    # logger.debug("logits_w_noise({})".format(logits_w_noise))  
    # logger.debug("logits_except1({})".format(logits_except1))  
    # logger.debug("mask2.shape({})".format(mask2.shape))  
    # logger.debug("mask2({})".format(mask2)) # 

    # Compute locations in capacity buffer
    locations1 = torch.cumsum(mask1, dim=0) - 1
    locations2 = torch.cumsum(mask2, dim=0) - 1
    # Update 2nd's location by accounting for locations of 1st
    locations2 += torch.sum(mask1, dim=0, keepdim=True)

    # logger.debug("locations1.shape({})".format(locations1.shape))
    # logger.debug("locations1({})".format(locations1))
    # logger.debug("locations2({})".format(locations2))
    
    # gating decisions
    exp_counts = torch.sum(mask1 + mask2, dim=0).detach().to('cpu')
    # logger.debug("exp1_counts({})".format(exp_counts))
    # exp_counts_2 = torch.sum(mask2, dim=0).detach().to('cpu')
    # logger.debug("exp2_counts({})".format(exp_counts_2))

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    # ce = torch.mean(mask1.float(), dim=0)
    ce = torch.mean((mask1 + mask2).float(), dim=0)
    # l_aux = torch.mean(me * ce) * num_experts * num_experts
    l_aux = torch.mean(me * ce) * num_experts * num_experts * 1/2
    
    # print("l_aux", l_aux)
    # logger.debug("ce2:{}".format(ce2))
    # logger.debug("new_mask:{}".format(new_mask))
    # print("new_l_aux", new_l_aux)

    # Remove locations outside capacity from mask
    mask1 *= torch.lt(locations1, capacity)
    mask2 *= torch.lt(locations2, capacity)

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)

    # Normalize gate probabilities
    mask1_float = mask1.float()
    mask2_float = mask2.float()
    gates1_s = einsum("se,se->s", gates, mask1_float) # top1 gate选择的tokens的权重
    gates2_s = einsum("se,se->s", gates, mask2_float) # top2 gate选择的tokens的权重
    denom_s = gates1_s + gates2_s
    
    # logger.debug("mask1_float:{}".format(mask1_float))
    # logger.debug("gates:{}".format(gates))
    # logger.debug("gates1_s:{}".format(gates1_s))
    # logger.debug("gates2_s:{}".format(gates2_s))
    
    # Avoid divide-by-zero
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    gates1_s /= denom_s
    gates2_s /= denom_s
    # logger.debug("gates1_s_new:{}".format(gates1_s))
    # logger.debug("gates2_s_new:{}".format(gates2_s))

    # Calculate combine_weights and dispatch_mask
    gates1 = einsum("s,se->se", gates1_s, mask1_float)
    gates2 = einsum("s,se->se", gates2_s, mask2_float)
    # logger.debug("gates1:{}".format(gates1)) # 8,4
    locations1_sc = _one_hot_to_float(locations1_s, capacity)
    locations2_sc = _one_hot_to_float(locations2_s, capacity)
    # logger.debug("locations1_sc:{}".format(locations1_sc)) # 8,4
    combine1_sec = einsum("se,sc->sec", gates1, locations1_sc)
    combine2_sec = einsum("se,sc->sec", gates2, locations2_sc)
    combine_weights = combine1_sec + combine2_sec
    # logger.debug("combine_weights:{}".format(combine_weights)) # 8,4,4
    dispatch_mask = combine_weights.bool()

    return l_aux, combine_weights, dispatch_mask, exp_counts


def dynamicgating(logits: Tensor,
               capacity_factor: float,
               min_capacity: int,
               threshold) -> Tuple[Tensor,
                                           Tensor,
                                           Tensor,
                                           Tensor]:
    """Implements Top2Gating on logits."""
    # everything is in fp32 in this function
    gates = F.softmax(logits, dim=1)

    capacity = _capacity(gates,
                         torch.tensor(capacity_factor * 2),
                         torch.tensor(min_capacity))
    # print(f" current capacity is: {capacity}")
    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1)
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    # Create a mask for 2nd's expert per token using Gumbel-max trick
    # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
    logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    # Replace top-expert with min value
    logits_except1 = logits_w_noise.masked_fill(mask1.bool(), float("-inf"))
    indices2_s = torch.argmax(logits_except1, dim=1)
    mask2 = F.one_hot(indices2_s, num_classes=num_experts)

    # logger.debug("logits:{}".format(logits))
    # logger.debug("mask1:{}".format(mask1))
    # logger.debug("mask2:{}".format(mask2))
    
    # get the top1 & top2 logits value
    top1_val = gates.masked_select(mask1.bool())
    top2_val = gates.masked_select(mask2.bool())
    dynamic_mask = (abs(top1_val - top2_val) > threshold).unsqueeze(1) # larger than threshold -> choose top1 gate -> remove dispatch entry from mask2
    # logger.debug("mean top1-top2:{}".format(torch.mean(top1_val - top2_val)))
    top1_selectednum = dynamic_mask.sum()
    top2_selectednum = logits.size(0) - top1_selectednum
    mask2 = mask2.masked_fill(dynamic_mask, False) # remask the dispatch tokens
    mean_threshold = torch.mean(top1_val - top2_val).detach()
    
    # logger.debug("top1_val:{}".format(top1_val))
    # logger.debug("top2_val:{}".format(top2_val))
    # logger.debug("dynamic_mask:{}".format(dynamic_mask))
    # logger.debug("top2_selectednum:{}".format(top2_selectednum))
    # logger.debug("top1_selectednum:{}".format(top1_selectednum))
    # logger.debug("mask2:{}".format(mask2))
    
    # Compute locations in capacity buffer
    locations1 = torch.cumsum(mask1, dim=0) - 1
    locations2 = torch.cumsum(mask2, dim=0) - 1
    # Update 2nd's location by accounting for locations of 1st
    locations2 += torch.sum(mask1, dim=0, keepdim=True)

    # gating decisions
    exp_counts = torch.sum(mask1 + mask2, dim=0).detach().to('cpu')  # 不受capacity限制的发送数目

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    # ce = torch.mean(mask1.float(), dim=0)
    ce = torch.mean((mask1 + mask2).float(), dim=0)
    # l_aux = torch.mean(me * ce) * num_experts * num_experts
    l_aux = torch.mean(me * ce) * num_experts * num_experts * 1/2
    
    # Remove locations outside capacity from mask
    mask1 *= torch.lt(locations1, capacity)
    mask2 *= torch.lt(locations2, capacity)

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)

    # Normalize gate probabilities
    mask1_float = mask1.float()
    mask2_float = mask2.float()
    gates1_s = einsum("se,se->s", gates, mask1_float) # top1 gate选择的tokens的权重
    gates2_s = einsum("se,se->s", gates, mask2_float) # top2 gate选择的tokens的权重
    denom_s = gates1_s + gates2_s
    # logger.debug("gates1_s:{}".format(gates1_s)) # 8,4,4
    # logger.debug("gates2_s:{}".format(gates2_s)) # 8,4,4
    
    # Avoid divide-by-zero
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    gates1_s /= denom_s
    gates2_s /= denom_s
    # logger.debug("min=torch.finfo(denom_s.dtype).eps:{}".format(torch.finfo(denom_s.dtype).eps))
    # logger.debug("gates1_s_new:{}".format(gates1_s)) # 8,4,4
    # logger.debug("gates2_s_new:{}".format(gates2_s)) # 8,4,4

    # Calculate combine_weights and dispatch_mask
    gates1 = einsum("s,se->se", gates1_s, mask1_float)
    gates2 = einsum("s,se->se", gates2_s, mask2_float)
    # logger.debug("gates1:{}".format(gates1)) # 8,4,4
    # logger.debug("gates2:{}".format(gates2)) # 8,4,4
    # logger.debug("gates1:{}".format(gates1)) # 8,4
    locations1_sc = _one_hot_to_float(locations1_s, capacity)
    locations2_sc = _one_hot_to_float(locations2_s, capacity)
    # logger.debug("locations1_sc:{}".format(locations1_sc)) # 8,4
    combine1_sec = einsum("se,sc->sec", gates1, locations1_sc)
    combine2_sec = einsum("se,sc->sec", gates2, locations2_sc)
    # logger.debug("combine1_sec:{}".format(combine1_sec)) # 8,4,4
    # logger.debug("combine2_sec:{}".format(combine2_sec)) # 8,4,4
    combine_weights = combine1_sec + combine2_sec
    # logger.debug("combine_weights:{}".format(combine_weights)) # 8,4,4
    dispatch_mask = combine_weights.bool()
    # logger.debug("dispatch_mask:{}".format(dispatch_mask)) # 8,4,4

    return l_aux, combine_weights, dispatch_mask, (exp_counts, top1_selectednum, top2_selectednum), mean_threshold

class TopKGate(Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = TopKGate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    wg: torch.nn.Linear

    def __init__(self,
                 model_dim: int,
                 num_experts: int,
                 k: int = 1,
                 capacity_factor: float = 1.0,
                 eval_capacity_factor: float = 1.0,
                 min_capacity: int = 8,
                 noisy_gate_policy: Optional[str] = None,
                 drop_tokens: bool = True,
                 use_rts: bool = True,
                 dyna_threshold: float = 0.075) -> None:
        super().__init__()

        # Only top-1 and top-2 are supported at the moment.
        if k != 1 and k != 2 and k != 0:
            raise ValueError('Only top-1 & top-2 & dynamic gatings are supported.')
        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False).float()
        self.k = k
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.min_capacity = min_capacity
        self.noisy_gate_policy = noisy_gate_policy
        self.timers = SynchronizedWallClockTimer()
        self.wall_clock_breakdown = False
        self.gate_time = 0.0
        self.drop_tokens = drop_tokens
        self.use_rts = use_rts
        self.dyna_threshold = torch.nn.Parameter(torch.Tensor([dyna_threshold]), requires_grad = False) # for model restore

    def forward(
            self,
            input: torch.Tensor,
            used_token: torch.Tensor = None,
            use_tutel: bool = False) -> Tuple[Tensor,
                                              Tensor,
                                              Tensor]:  # type: ignore

        if self.wall_clock_breakdown:
            self.timers('TopKGate').start()

        if self.wg.weight.dtype != torch.float32:
            self.wg = self.wg.float()
        input_fp32 = input.float()
        # input jittering
        if self.noisy_gate_policy == 'Jitter' and self.training:
            input_fp32 = multiplicative_jitter(input_fp32, device=input.device)
        logits = self.wg(input_fp32)

        if self.k == 1:
            gate_output = top1gating(
                logits,
                self.capacity_factor if self.training else self.eval_capacity_factor,
                self.min_capacity,
                used_token,
                self.noisy_gate_policy if self.training else None,
                self.drop_tokens,
                self.use_rts,
                use_tutel)

        elif self.k == 2:
            gate_output = top2gating(
                logits,
                self.capacity_factor if self.training else self.eval_capacity_factor,
                self.min_capacity)
        else:
            gate_output = dynamicgating(
                logits,
                self.capacity_factor if self.training else self.eval_capacity_factor,
                self.min_capacity,
                threshold=self.dyna_threshold
                )
            self.dyna_threshold[0] = 0.95 * self.dyna_threshold + 0.04 * gate_output[4]
            gate_output = gate_output[0:4]
            

        if self.wall_clock_breakdown:
            self.timers('TopKGate').stop()
            self.gate_time = self.timers('TopKGate').elapsed(reset=False)

        return gate_output


class MOELayer(Base):
    """MOELayer module which implements MixtureOfExperts as described in Gshard_.
    ::

        gate = TopKGate(model_dim, num_experts)
        moe = MOELayer(gate, expert)
        output = moe(input)
        l_aux = moe.l_aux

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        gate (torch.nn.Module):
            gate network
        expert (torch.nn.Module):
            expert network
    """
    def __init__(self,
                 gate: Module,
                 experts: Module,
                 ep_group_name,
                 ep_size,
                 num_local_experts: int,
                 use_tutel: bool = False) -> None:
        super().__init__()
        self.gate = gate
        self.experts = experts
        self.ep_group = None
        self.ep_size = ep_size
        self.ep_group_name = ep_group_name
        self.num_local_experts = num_local_experts
        self.time_falltoall = 0.0
        self.time_salltoall = 0.0
        self.time_moe = 0.0
        self.timers = SynchronizedWallClockTimer()
        self.wall_clock_breakdown = False

        self.use_tutel = use_tutel and TUTEL_INSTALLED and gate.k == 1

        if self.use_tutel:
            logger.info('Using Tutel optimizations.')
        elif use_tutel and not TUTEL_INSTALLED:
            logger.warning("Tutel optimization requested but not installed. "
                           "Proceeding without Tutel.")
        elif use_tutel and TUTEL_INSTALLED and gate.k != 1:
            logger.warning(
                "To enable Tutel optimization, use top-1 instead of top-2 gate. "
                "Proceeding without Tutel.")

    def _set_ep_group(self, ep_group):
        self.ep_group = ep_group

    def forward(self, *input: Tensor, **kwargs: Any) -> Tensor:

        if self.wall_clock_breakdown:
            self.timers('moe').start()

        # Implement Algorithm 2 from GShard paper.
        d_model = input[0].shape[-1]
        # logger.debug("d_model:{}".format(d_model)) # 8

        # Initial implementation -> Reshape into S tokens by dropping sequence dimension.
        # Reshape into G groups so that each group can distribute tokens equally
        # group_size = kwargs['group_size'] if 'group_size' in kwargs.keys() else 1
        reshaped_input = input[0].reshape(-1, d_model)
        # print(f"RAW: rank:{dist.get_rank()}, reshaped_input:{reshaped_input}")


        if self.use_tutel:
            self.l_aux, C, E, indices_, locations_, gates_, self.exp_counts = self.gate(reshaped_input, input[1], True)
            S, M = reshaped_input.size(0), reshaped_input.size(1)

            if not hasattr(self, '_tutel_dispatcher'):
                self._tutel_dispatcher = tutel_moe.fast_dispatcher(
                    E,
                    C,
                    M,
                    dispatch_dtype=reshaped_input.dtype)
            self._tutel_dispatcher.update(indices_, locations_, gates_, capacity=C)
            dispatched_input = self._tutel_dispatcher.encode(reshaped_input)
        else:
            self.l_aux, combine_weights, dispatch_mask, self.exp_counts = self.gate(reshaped_input, input[1])
            # 通过top1 gate之后生成的一个dispatch_mask的具体形式是(num of tokens, num of experts, capacity)
            # 其中num of experts, capacity中，第n行代表分配给第n个expert的token情况，第n列代表当前token是capacity的第几个
            # 例如dispatch_mask为(8,2,4)的情形时：
            # [[False, False, False, False],    第一个token 不分配给expert1
            #  [ True, False, False, False]],                分配给expert2，占用其第一个capacity 1/4
            
            # [[ True, False, False, False],    第二个token 分配给expert 1，占用其第一个capacity 1/4
            #  [False, False, False, False]],               不分配给expert 2

            # [[False, False, False, False],    第三个token  
            #  [False,  True, False, False]],               分配给expert2，占用其第二个capacity 2/4
    
            # [[False,  True, False, False],    第三个token  分配给expert1，占用其第二个capacity 2/4
            #  [False, False, False, False]],
    
            # [[False, False, False, False],
            #  [False, False,  True, False]],
    
            # [[False, False,  True, False],
            #  [False, False, False, False]],
    
            # [[False, False, False, False],
            #  [False, False, False,  True]],
    
            # [[False, False, False,  True],
            #  [False, False, False, False]]
            # logger.debug("dispatch_mask.shape({}), dispatch_mask:{}".format(dispatch_mask.shape, dispatch_mask)) # 8, 2, 4
            # logger.debug("combine_weights.shape({}), combine_weights:{}".format(combine_weights.shape, combine_weights.type_as(input[0])))
            dispatched_input = einsum("sec,sm->ecm",
                                      dispatch_mask.type_as(input[0]),
                                      reshaped_input)  # 8, 84
            # 这里通过einsum和将mask与tokens相乘，得到最终分派给不同experts的具体tokens
            # 例如这里的 einsum (8, 2, 4) * (8, 84) -> (2, 4, 84) 就是用8个(2,4)矩阵里的每一个元素与(8, 84)矩阵每行的元素相乘 -> 一共得到84列矩阵
            # 在mask的8个(2, 4)的矩阵联系起来，(1,1,1) (2,1,1) ... (8,1,1)连起来，代表了dispatch到第一个expert的capacity为1/4的具体token是什么
            # logger.debug("dispatched_input.shape({}), dispatched_input:{}".format(dispatched_input.shape, dispatched_input)) # 2, 4, 84
        
        # print(f"RAW: rank:{dist.get_rank()}, combine_weights:{combine_weights}")

        curr_exp_counts = self.exp_counts[0] if type(self.exp_counts) == tuple else self.exp_counts
        # print(f"RAW: rank:{dist.get_rank()}, curr_exp_dispatch_counts:{curr_exp_counts}")

        if self.wall_clock_breakdown:
            self.timers('falltoall').start()

        if groups._get_expert_model_parallel_world_size() == 1:
            # If the non-expert is tensor-parallel, it will create
            # duplicate tokens on the tensor-parallel ranks.
            # Since our experts are not tensor-parallel, these duplicates
            # need to be dropped to ensure correctness.
            # this also doubles up as a communication optimization as we are
            # reducing the all-to-all communication volume.
            dispatched_input = drop_tokens(dispatched_input, dim=1)
            # logger.debug("groups._get_expert_model_parallel_world_size({})".format(groups._get_expert_model_parallel_world_size())) # 2, 4, 84
            # logger.debug("Drop tokens -> dispatched_input.shape({}), \n dispatched_input:{}".format(dispatched_input.shape, dispatched_input)) # 2, 4, 84
            
        # logger.debug(f"Before all2all dispatched_input.shape({dispatched_input.shape}) data:{dispatched_input}") # 2, 4, 84
        dispatched_input = _AllToAll.apply(self.ep_group, dispatched_input)
        # logger.debug(f"After all2all dispatched_input.shape({dispatched_input.shape}) data:{dispatched_input}") # 2, 4, 84

        if self.wall_clock_breakdown:
            self.timers('falltoall').stop()
            self.time_falltoall = self.timers('falltoall').elapsed(reset=False)

        # Re-shape after all-to-all: ecm -> gecm
        dispatched_input = dispatched_input.reshape(self.ep_size,
                                                    self.num_local_experts,
                                                    -1,
                                                    d_model)
        # print(f"RAW: rank:{dist.get_rank()}, dispatched_input tokens:{(dispatched_input)}")
        expert_output = self.experts(dispatched_input)

        if self.wall_clock_breakdown:
            self.timers('salltoall').start()

        expert_output = _AllToAll.apply(self.ep_group, expert_output)
        # print(f"RAW: rank:{dist.get_rank()}, expert_output:{expert_output}")

        if self.wall_clock_breakdown:
            self.timers('salltoall').stop()
            self.time_salltoall = self.timers('salltoall').elapsed(reset=False)

        # Re-shape back: gecm -> ecm
        expert_output = expert_output.reshape(self.ep_size * self.num_local_experts,
                                              -1,
                                              d_model)

        if groups._get_expert_model_parallel_world_size() == 1:
            # the dropped duplicate tokens need to be gathered on each
            # tensor parallel rank again for the tensor-parallel
            # non-expert of the next layer.
            expert_output = gather_tokens(expert_output, dim=1)

        # logger.debug("expert_ouput.shape({}), expert_output:{}".format(expert_output.shape, expert_output))
        # enisum: (8, 2, 4), (2, 4, 84) -> (8, 84)
        # combine_weights是sparse的，在(2,4)中，只有一个元素非0，其他都是false
        # einsum计算的结果为当前rank的tokens在expert计算结果 * 相应weight
        if self.use_tutel:
            combined_output = self._tutel_dispatcher.decode(expert_output.view(E * C, M))
        else:
            combined_output = einsum("sec,ecm->sm",
                                     combine_weights.type_as(input[0]),
                                     expert_output)

        # logger.debug("combined_output.shape({}), combined_output:{}".format(combined_output.shape, combined_output))
        a = combined_output.reshape(input[0].shape)
        # print(f"RAW: rank:{dist.get_rank()}, weighted_sum:{a}")
        
        if self.wall_clock_breakdown:
            self.timers('moe').stop()
            self.time_moe = self.timers('moe').elapsed(reset=False)

        # exit(0)
        return a


class DynamicMOELayer(Base):
    """MOELayer module which implements MixtureOfExperts as described in Gshard_.
    ::

        gate = TopKGate(model_dim, num_experts)
        moe = MOELayer(gate, expert)
        output = moe(input)
        l_aux = moe.l_aux

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        gate (torch.nn.Module):
            gate network
        expert (torch.nn.Module):
            expert network
    """
    def __init__(self,
                 gate: Module,
                 experts: Module,
                 ep_group_name,
                 ep_size,
                 num_local_experts: int,
                 num_exp_replica: int,
                 current_experts,
                 current_intra_node_placement,
                 use_tutel: bool = False) -> None:
        super().__init__()
        self.gate = gate
        self.experts = experts
        self.ep_group = None
        self.ep_size = ep_size
        self.ep_group_name = ep_group_name
        self.num_local_experts = num_local_experts # num_exp_replica + unique experts
        self.num_exp_replica = num_exp_replica
        self.current_experts_indices = current_experts.tolist() # current experts indices
        self.time_falltoall = 0.0
        self.time_salltoall = 0.0
        self.time_moe = 0.0
        self.timers = SynchronizedWallClockTimer()
        self.wall_clock_breakdown = False
        self.current_intra_node_placement=current_intra_node_placement # current intra-node placement info

        self.use_tutel = use_tutel and TUTEL_INSTALLED and gate.k == 1

        if self.use_tutel:
            logger.info('Using Tutel optimizations.')
        elif use_tutel and not TUTEL_INSTALLED:
            logger.warning("Tutel optimization requested but not installed. "
                           "Proceeding without Tutel.")
        elif use_tutel and TUTEL_INSTALLED and gate.k != 1:
            logger.warning(
                "To enable Tutel optimization, use top-1 instead of top-2 gate. "
                "Proceeding without Tutel.")

    def _set_ep_group(self, ep_group):
        self.ep_group = ep_group

    def all_to_all_exchange_size(
            self,
            group: torch.distributed.ProcessGroup,
            input: Tensor) -> Tensor:  # type: ignore
        input = input.contiguous()
        output = torch.empty_like(input)
        dist.all_to_all_single(output, input, group=group)
        return output
    
    def single_round(self, 
                     group: Any,
                     send_tokens_idx:list,
                     input_split: list, 
                     tokens: Tensor,
                     d_model: int,
                     _current_capacity,
                     recv_experts_outputs: list,
                     expert_index: int
                     ):
        """single round all2all communication and expert computation

        Args:
            group (Any): comm group
            send_tokens_idx (list): indices to place experts outputs (intra node only)
            input_split (list): input split of tokens (intra/inter all2all comm)
            tokens (Tensor): input tokens (intra/inter all2all comm)
            d_model (int): dimension of model
            _current_capacity (_type_): capacity (for padding)
            recv_experts_outputs (list): experts outputs (after all2all comm)
            expert_index (int): index of expert in computation
        """
        device = torch.cuda.current_device()
        # exchange size
        exchange_size=torch.tensor(input_split, device=device, dtype=torch.int32)
        output_split = self.all_to_all_exchange_size(group, exchange_size).tolist()
        # exchange tokens
        flatten_send_tokens = flatten(tokens)
        recv_expert_tokens = _AllToAll_UNEQUAL.apply(group, flatten_send_tokens,
                                                input_split, output_split)
        curr_expert_tokens = recv_expert_tokens.reshape((-1, d_model))
        output = self.experts(curr_expert_tokens, expert_index)
        # output.register_hook(lambda grad: print(f"rank:{dist.get_rank()}, expert {self.experts.curr_name[expert_index]} calc finish"))
        flatten_results = output.flatten()
        # send results back
        expert_outputs = _AllToAll_UNEQUAL.apply(group, flatten_results,
                                                output_split,input_split)
        # expert_outputs.register_hook(lambda grad: print(f"rank:{dist.get_rank()}, backward all2all expert_outputs:{expert_outputs.shape}"))
        
        for i, sync in enumerate(unflatten(expert_outputs, tokens)):
            # calc output placement index
            output_index = send_tokens_idx[i]
            # if processing world all2all, skip placing results of local experts
            if recv_experts_outputs[output_index] != []: continue
            # padding for einsum
            if sync.shape[0]==0: sync = sync.reshape((0, d_model))
            sync = torch.nn.functional.pad(
                sync, 
                (0,0,0,_current_capacity - sync.shape[0]), 
                'constant', 
                0)
            assert sync.shape[0] == _current_capacity
            # place the output in an appropriate place
            recv_experts_outputs[output_index] = sync
        
    def inter_round(self, 
                     group: Any,
                     send_tokens_idx:list,
                     input_split: list, 
                     tokens: Tensor,
                     d_model: int,
                     _current_capacity,
                     recv_experts_outputs: list,
                     current_experts_indices: list,
                     num_local_unique_experts: int
                     ):
        """single round all2all communication and expert computation

        Args:
            group (Any): comm group
            send_tokens_idx (list): indices to place experts outputs (intra node only)
            input_split (list): input split of tokens (intra/inter all2all comm)
            tokens (Tensor): input tokens (intra/inter all2all comm)
            d_model (int): dimension of model
            _current_capacity (_type_): capacity (for padding)
            recv_experts_outputs (list): experts outputs (after all2all comm)
            expert_index (int): index of expert in computation
        """
        device = torch.cuda.current_device()
        # exchange size
        exchange_size=torch.tensor(input_split, device=device, dtype=torch.int32)
        output_split = self.all_to_all_exchange_size(group, exchange_size).tolist()
        _combined_input_split=[]
        _combined_output_split=[]
        for i in range(0, len(input_split), num_local_unique_experts):
            _combined_input_split.append (sum(input_split [i: i+num_local_unique_experts]))
            _combined_output_split.append(sum(output_split[i: i+num_local_unique_experts]))
            
        # exchange tokens
        flatten_send_tokens = flatten(tokens)
        recv_expert_tokens = _AllToAll_UNEQUAL.apply(group, flatten_send_tokens,
                                                _combined_input_split, _combined_output_split)
        curr_expert_tokens = recv_expert_tokens.reshape((-1, d_model))

        _output_split_s = [o//d_model for o in output_split]
        _recv_tokens_split= list(torch.split(curr_expert_tokens, split_size_or_sections=_output_split_s, dim=0))
        
        output_send_back=[[] for _ in range(len(output_split))]
        for i in range(num_local_unique_experts):
            # get tokens for one expert
            _tokens_2_experts = _recv_tokens_split[i::num_local_unique_experts]
            # get size of the tokens in calculation
            _size = _output_split_s[i::num_local_unique_experts]
            # concatenate the tokens 
            _tokens_in_use = torch.cat(_tokens_2_experts, dim=0)
            # calc expert index in calculation
            expert_index = current_experts_indices.index(dist.get_rank()*num_local_unique_experts + i)
            # expert calculate results
            _expert_output = self.experts(_tokens_in_use, expert_index)
            # split back
            expert_output_split = torch.split(_expert_output, split_size_or_sections = _size, dim=0)
            # put elements back in tensor list
            output_send_back[i::num_local_unique_experts] = expert_output_split
        
        # output = self.experts(curr_expert_tokens, expert_index)
        # output.register_hook(lambda grad: print(f"rank:{dist.get_rank()}, expert {self.experts.curr_name[expert_index]} calc finish"))
        flatten_results = flatten(output_send_back)
        # send results back
        expert_outputs = _AllToAll_UNEQUAL.apply(group, flatten_results,
                                                _combined_output_split, _combined_input_split)
        # expert_outputs.register_hook(lambda grad: print(f"rank:{dist.get_rank()}, backward all2all expert_outputs:{expert_outputs.shape}"))
        
        for i, sync in enumerate(unflatten(expert_outputs, tokens)):
            # calc output placement index
            output_index = send_tokens_idx[i]
            # if processing world all2all, skip placing results of local experts
            if recv_experts_outputs[output_index] != []: continue
            # padding for einsum
            if sync.shape[0]==0: sync = sync.reshape((0, d_model))
            sync = torch.nn.functional.pad(
                sync, 
                (0,0,0,_current_capacity - sync.shape[0]), 
                'constant', 
                0)
            assert sync.shape[0] == _current_capacity
            # place the output in an appropriate place
            recv_experts_outputs[output_index] = sync

    def forward(self, *input: Tensor, **kwargs: Any) -> Tensor:
        
        intra_comm_group = groups._get_dynamic_expert_all_to_all_group()
        device = torch.cuda.current_device()
        dtype=input[0].dtype

        if self.wall_clock_breakdown:
            self.timers('moe').start()

        # Implement Algorithm 2 from GShard paper.
        d_model = input[0].shape[-1]
        # logger.debug("d_model:{}".format(d_model)) # 8

        # Initial implementation -> Reshape into S tokens by dropping sequence dimension.
        # Reshape into G groups so that each group can distribute tokens equally
        # group_size = kwargs['group_size'] if 'group_size' in kwargs.keys() else 1
        reshaped_input = input[0].reshape(-1, d_model)
        # print(f"DYNA: rank:{dist.get_rank()}, reshaped_input:{reshaped_input}")

        if self.use_tutel:
            self.l_aux, C, E, indices_, locations_, gates_, self.exp_counts = self.gate(reshaped_input, input[1], True)
            S, M = reshaped_input.size(0), reshaped_input.size(1)

            if not hasattr(self, '_tutel_dispatcher'):
                self._tutel_dispatcher = tutel_moe.fast_dispatcher(
                    E,
                    C,
                    M,
                    dispatch_dtype=reshaped_input.dtype)
            self._tutel_dispatcher.update(indices_, locations_, gates_, capacity=C)
            dispatched_input = self._tutel_dispatcher.encode(reshaped_input)
        else:
            self.l_aux, combine_weights, dispatch_mask, self.exp_counts = self.gate(reshaped_input, input[1])
            # 通过top1 gate之后生成的一个dispatch_mask的具体形式是(num of tokens, num of experts, capacity)
            # 其中num of experts, capacity中，第n行代表分配给第n个expert的token情况，第n列代表当前token是capacity的第几个
            dispatched_input = einsum("sec,sm->ecm", # (num of experts, capacity, dimension)
                                      dispatch_mask.type_as(input[0]),
                                      reshaped_input)  # 8, 84
            # 这里通过einsum和将mask与tokens相乘，得到最终分派给不同experts的具体tokens
            # 例如这里的 einsum (8, 2, 4) * (8, 84) -> (2, 4, 84) 就是用8个(2,4)矩阵里的每一个元素与(8, 84)矩阵每行的元素相乘 -> 一共得到84列矩阵
            # 在mask的8个(2, 4)的矩阵联系起来，(1,1,1) (2,1,1) ... (8,1,1)连起来，代表了dispatch到第一个expert的capacity为1/4的具体token是什么

            # print(f"DYNA: rank:{dist.get_rank()}, dispatched_input:{dispatched_input}")
            _current_capacity = dispatched_input.shape[1]
            chunks = dispatched_input.chunk(dispatched_input.shape[0], dim=0)

            curr_exp_counts = self.exp_counts[0] if type(self.exp_counts) == tuple else self.exp_counts
            # logger.debug(f"DYNA: rank:{dist.get_rank()}, curr_exp_dispatch_counts:{curr_exp_counts}")
            
            # get tokens to be sent
            tokens = []
            for i, send in enumerate(curr_exp_counts):
                # TODO: Try to optimize einsum calculation
                # set send == 1 can avoid backward propagation deadlock in world all2all communication
                # if the tokens dispatch to a specified expert is 0, then backward computation in `single_round` function
                # may suffer deadlock as the padding 0 block the chain rule of tensor autograder (block propagation)
                if send==0: send = 1 
                tokens.append(chunks[i].squeeze(dim=0)[0:send])

            ''' first round: intra node '''
            # {num_local_experts} round all2all
            _processed_idx=[]
            recv_experts_outputs = [[] for _ in range(dispatched_input.shape[0])]
            for i in range(self.num_local_experts):
                send_tokens_idx = []
                for j in range(topology._get_gpu_per_node_number()):
                    send_tokens_idx.append(self.current_intra_node_placement[j][i])
                _processed_idx.extend(send_tokens_idx)
                
                token_send=[]
                input_split=[]
                for idx in send_tokens_idx:
                    token_send.append(tokens[idx])
                    # tokens[idx].register_hook(lambda grad: print(f"rank:{dist.get_rank()}, index:{idx}, intra-grad {grad.shape} calc finish"))
                    input_split.append(tokens[idx].numel())
                self.single_round(group=intra_comm_group,
                                  send_tokens_idx=send_tokens_idx,
                                  input_split=input_split,
                                  tokens=token_send,
                                  d_model=d_model,
                                  _current_capacity=_current_capacity,
                                  recv_experts_outputs=recv_experts_outputs,
                                  expert_index=i)
            
            ''' second round: inter node '''
            # TODO: rewrite global communication, in case of global exchange tokens with size [0,0,0,0]
            num_local_unique_experts = self.num_local_experts - self.num_exp_replica
            send_tokens_idx = []
            global_token_send=[]
            global_token_send_split=[]
            for i in range(dist.get_world_size()):
                _tokens_global=[]
                _tokens_global_size=[]
                for idx in range(i * num_local_unique_experts, (i+1) * num_local_unique_experts):
                    send_tokens_idx.append(idx)
                    if idx not in _processed_idx:
                        _tokens_global.append(tokens[idx])
                        _tokens_global_size.append(tokens[idx].numel())
                        # global_input_split.append(tokens[idx].numel())
                        _processed_idx.append(idx)
                    else:
                        _send_empty=torch.tensor([], dtype=dtype, device=device)
                        _tokens_global.append(_send_empty)
                        _tokens_global_size.append(0)
                global_token_send.extend(_tokens_global)
                global_token_send_split.extend(_tokens_global_size)

            self.inter_round(group=None,
                send_tokens_idx=send_tokens_idx,
                input_split=global_token_send_split,
                tokens=global_token_send,
                d_model=d_model,
                _current_capacity=_current_capacity,
                recv_experts_outputs=recv_experts_outputs,
                current_experts_indices=self.current_experts_indices,
                num_local_unique_experts=num_local_unique_experts,
            )
                        # global_token_send.append(_send_empty)
                        # global_input_split.append(0)
                        # tokens[idx].register_hook(lambda grad: print(f"rank:{dist.get_rank()}, index:{idx}, inter-grad {grad.shape} finish"))
            
            # original code
            # num_local_unique_experts = self.num_local_experts - self.num_exp_replica
            # for i in range(num_local_unique_experts):
            #     send_tokens_idx = []
            #     global_token_send=[]
            #     global_input_split = []
            #     for idx in range(i, dispatched_input.shape[0], num_local_unique_experts):
            #         send_tokens_idx.append(idx)
            #         if idx not in _processed_idx:
            #             global_token_send.append(tokens[idx])
            #             global_input_split.append(tokens[idx].numel())
            #             _processed_idx.append(idx)
            #             tokens[idx].register_hook(lambda grad: print(f"rank:{dist.get_rank()}, index:{idx}, inter-grad {grad.shape} finish"))
                        
            #         else:
            #             _send_empty=torch.tensor([], dtype=dtype, device=device)
            #             global_token_send.append(_send_empty)
            #             global_input_split.append(0)

            #     print(f"**** inter_all_to_all, input_split:{global_input_split} token send by expert{self.experts.curr_name[self.current_experts_indices.index(dist.get_rank()*num_local_unique_experts + i)]} ****")
            #     self.single_round(group=None,
            #                 send_tokens_idx=send_tokens_idx,
            #                 input_split=global_input_split,
            #                 tokens=global_token_send,
            #                 d_model=d_model,
            #                 _current_capacity=_current_capacity,
            #                 recv_experts_outputs=recv_experts_outputs,
            #                 expert_index=self.current_experts_indices.index(dist.get_rank()*num_local_unique_experts + i))
        
        if self.wall_clock_breakdown:
            self.timers('falltoall').start()

        if groups._get_expert_model_parallel_world_size() == 1:
            # If the non-expert is tensor-parallel, it will create
            # duplicate tokens on the tensor-parallel ranks.
            # Since our experts are not tensor-parallel, these duplicates
            # need to be dropped to ensure correctness.
            # this also doubles up as a communication optimization as we are
            # reducing the all-to-all communication volume.
            dispatched_input = drop_tokens(dispatched_input, dim=1)
            # logger.debug("groups._get_expert_model_parallel_world_size({})".format(groups._get_expert_model_parallel_world_size())) # 2, 4, 84
            # logger.debug("Drop tokens -> dispatched_input.shape({}), \n dispatched_input:{}".format(dispatched_input.shape, dispatched_input)) # 2, 4, 84
            
        if self.wall_clock_breakdown:
            self.timers('falltoall').stop()
            self.time_falltoall = self.timers('falltoall').elapsed(reset=False)

        if self.wall_clock_breakdown:
            self.timers('salltoall').start()

        if self.wall_clock_breakdown:
            self.timers('salltoall').stop()
            self.time_salltoall = self.timers('salltoall').elapsed(reset=False)

        # Re-shape back: gecm -> ecm        
        # print(f"DYNA: rank:{dist.get_rank()}, recv_experts_outputs:{recv_experts_outputs}")
        expert_output = torch.cat(recv_experts_outputs, dim=0) # 横向拼接
        expert_output = expert_output.reshape(self.ep_size * (self.num_local_experts - self.num_exp_replica),
                                              -1,
                                              d_model)
        # expert_output.register_hook(lambda grad: print(f"rank:{dist.get_rank()}, expert_output grad {grad.shape} calc finish"))
        
        
        if groups._get_expert_model_parallel_world_size() == 1:
            # the dropped duplicate tokens need to be gathered on each
            # tensor parallel rank again for the tensor-parallel
            # non-expert of the next layer.
            expert_output = gather_tokens(expert_output, dim=1)

        if self.use_tutel:
            combined_output = self._tutel_dispatcher.decode(expert_output.view(E * C, M))
        else:
            combined_output = einsum("sec,ecm->sm",
                                     combine_weights.type_as(input[0]),
                                     expert_output)

        # logger.debug("combined_output.shape({}), combined_output:{}".format(combined_output.shape, combined_output))
        a = combined_output.reshape(input[0].shape)
        # logger.debug("a.shape({}), a:{}".format(a.shape, a))
        # print(f"DYNA: rank:{dist.get_rank()}, weighted_sum:{a}")

        if self.wall_clock_breakdown:
            self.timers('moe').stop()
            self.time_moe = self.timers('moe').elapsed(reset=False)

        return a

