'''
Copyright 2020 The Microsoft DeepSpeed Team
'''

from deepspeed.utils import logger
import torch
import copy


class Experts(torch.nn.Module):
    def __init__(self, expert, num_local_experts=1, expert_group_name=None):
        super(Experts, self).__init__()

        self.deepspeed_experts = torch.nn.ModuleList(
            [copy.deepcopy(expert) for i in range(num_local_experts)])
        self.num_local_experts = num_local_experts

        # TODO: revisit allreduce for moe.gate...
        for expert in self.deepspeed_experts:
            # TODO: Create param groups to handle expert + data case (e.g. param.group = moe_group)
            for name, param in expert.named_parameters():
                param.allreduce = False
                param.group_name = expert_group_name

    def forward(self, inputs):
        # logger.debug("experts_inputs.shape({})".format(inputs.shape))
        chunks = inputs.chunk(self.num_local_experts, dim=1)
        # logger.debug("length_chunks:{} experts_chunks.shape({})".format(len(chunks), chunks[0].shape))
        expert_outputs = []
        for chunk, expert in zip(chunks, self.deepspeed_experts): # 在all2all之后，第i个chunk中的内容，由第i个expert计算
            out = expert(chunk)
            if type(out) is tuple:
                out = out[0]  # Ignore the bias term for now
            expert_outputs += [out]

        expert_output = torch.cat(expert_outputs, dim=1) # 横向拼接
        return expert_output


class DynamicExperts(torch.nn.Module):
    def __init__(self, expert, num_local_experts=1, expert_group_name=None, curr_name=None):
        super(DynamicExperts, self).__init__()

        self.deepspeed_experts = torch.nn.ModuleList(
            [copy.deepcopy(expert) for i in range(num_local_experts)])
        self.num_local_experts = num_local_experts

        for i, expert in enumerate(self.deepspeed_experts):
            
            for name, param in expert.named_parameters():
                param.allreduce = False
                param.group_name = curr_name[i]
                # param.expert_name=curr_name[i]

    def forward(self, inputs):
        # logger.debug("experts_inputs.shape({})".format(inputs.shape))
        chunks = inputs.chunk(self.num_local_experts, dim=1)
        # logger.debug("length_chunks:{} experts_chunks.shape({})".format(len(chunks), chunks[0].shape))
        expert_outputs = []
        for chunk, expert in zip(chunks, self.deepspeed_experts): # 在all2all之后，第i个chunk中的内容，由第i个expert计算
            out = expert(chunk)
            if type(out) is tuple:
                out = out[0]  # Ignore the bias term for now
            expert_outputs += [out]

        expert_output = torch.cat(expert_outputs, dim=1) # 横向拼接
        return expert_output