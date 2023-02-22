'''
Copyright 2020 The Microsoft DeepSpeed Team
'''

import os
import torch
import numpy as np
import torch.distributed as dist
from deepspeed.utils import log_dist

from deepspeed.utils import groups, topology
from .sharded_moe import MOELayer, TopKGate, DynamicMOELayer
from .experts import Experts, DynamicExperts
import typing


class MoE(torch.nn.Module):
    def __init__(self,
                 hidden_size,
                 expert,
                 num_experts=1,
                 ep_size=1,
                 k=1,
                 capacity_factor=1.,
                 eval_capacity_factor=1.,
                 min_capacity=4,
                 use_residual=False,
                 noisy_gate_policy: typing.Optional[str] = None,
                 drop_tokens: bool = True,
                 use_rts=True,
                 use_tutel: bool = False,
                 enable_expert_tensor_parallelism: bool = False,
                 dyna_threshold: float = 0.015):
        """Initialize an MoE layer.

        Arguments:
            hidden_size (int): the hidden dimension of the model, importantly this is also the input and output dimension.
            expert (torch.nn.Module): the torch module that defines the expert (e.g., MLP, torch.linear).
            num_experts (int, optional): default=1, the total number of experts per layer.
            ep_size (int, optional): default=1, number of ranks in the expert parallel world or group.
            k (int, optional): default=1, top-k gating value, only supports k=1 or k=2.
            capacity_factor (float, optional): default=1.0, the capacity of the expert at training time.
            eval_capacity_factor (float, optional): default=1.0, the capacity of the expert at eval time.
            min_capacity (int, optional): default=4, the minimum capacity per expert regardless of the capacity_factor.
            use_residual (bool, optional): default=False, make this MoE layer a Residual MoE (https://arxiv.org/abs/2201.05596) layer.
            noisy_gate_policy (str, optional): default=None, noisy gate policy, valid options are 'Jitter', 'RSample' or 'None'.
            drop_tokens (bool, optional): default=True, whether to drop tokens - (setting to False is equivalent to infinite capacity).
            use_rts (bool, optional): default=True, whether to use Random Token Selection.
            use_tutel (bool, optional): default=False, whether to use Tutel optimizations (if installed).
            enable_expert_tensor_parallelism (bool, optional): default=False, whether to use tensor parallelism for experts
        """

        super(MoE, self).__init__()

        self.use_residual = use_residual
        self.enable_expert_tensor_parallelism = enable_expert_tensor_parallelism
        assert num_experts % ep_size == 0, f"Number of experts ({num_experts}) should be divisible by expert parallel size ({ep_size})"
        self.ep_size = ep_size # experts并行group的总GPU数量
        self.expert_group_name = f"ep_size_{self.ep_size}"
        self.num_experts = num_experts
        self.num_local_experts = num_experts // self.ep_size # 每个gpu上local_experts个数

        log_dist(
            f'Creating MoE layer with num_experts: {num_experts} | num_local_experts: {self.num_local_experts} | expert_parallel_size: {self.ep_size}',
            [0])

        assert noisy_gate_policy is None or noisy_gate_policy in ['None', 'Jitter', 'RSample'], \
            'Unsupported noisy_gate_policy: ' + noisy_gate_policy

        experts = Experts(expert, self.num_local_experts, self.expert_group_name)
        self.deepspeed_moe = MOELayer(TopKGate(hidden_size,
                                               num_experts,
                                               k,
                                               capacity_factor,
                                               eval_capacity_factor,
                                               min_capacity,
                                               noisy_gate_policy,
                                               drop_tokens,
                                               use_rts,
                                               dyna_threshold),
                                      experts,
                                      self.expert_group_name,
                                      self.ep_size,
                                      self.num_local_experts,
                                      use_tutel=use_tutel)
        if self.use_residual:
            self.mlp = expert
            # coefficient is used for weighted sum of the output of expert and mlp
            self.coefficient = torch.nn.Linear(hidden_size, 2)

    def set_deepspeed_parallelism(self):
        self._create_process_groups()

    def _create_process_groups(self):
        # Create process group for a layer if needed
        if self.expert_group_name not in groups._get_expert_parallel_group_dict():
            print(
                f"No existing process group found, creating a new group named: {self.expert_group_name}"
            )
            if (groups.mpu is None) or (not self.enable_expert_tensor_parallelism):
                # Condition 1 - no groups.mpu means no tensor parallelism
                # Condition 2 - disabling expert tensor parallelism on purpose
                groups._create_expert_and_data_parallel(self.ep_size)
            else:
                # expert tensor parallelism is enabled
                groups._create_expert_data_and_model_parallel(self.ep_size,
                                                              mpu=groups.mpu)
        # Set the group handle for the MOELayer (deepspeed_moe) object
        self.deepspeed_moe._set_ep_group(
            groups._get_expert_parallel_group(self.expert_group_name))

    def forward(self, hidden_states, used_token=None):
        """ MoE forward

        Arguments:
            hidden_states (Tensor): input to the layer
            used_token (Tensor, optional): default: None, mask only used tokens

        Returns:
            A tuple including output, gate loss, and expert count.

            * output (Tensor): output of the model

            * l_aux (Tensor): gate loss value

            * exp_counts (int): expert count
        """
        output = self.deepspeed_moe(hidden_states, used_token)
        if self.use_residual:
            # Residual MoE
            output_mlp = self.mlp(hidden_states)
            if type(output_mlp) is tuple:
                output_mlp = output_mlp[0]  # Ignore the bias term for now
            coef = self.coefficient(hidden_states)
            coef = torch.nn.functional.softmax(coef, dim=-1)
            output = output * coef[..., 0:1] + output_mlp * coef[..., 1:]
        return output, self.deepspeed_moe.l_aux, self.deepspeed_moe.exp_counts


    
        
class DynamicMoE(torch.nn.Module):
    def __init__(self,
                 hidden_size,
                 expert,
                 layer_idx,
                 num_experts=1,
                 ep_size=1,
                 k=1,
                 capacity_factor=1.,
                 eval_capacity_factor=1.,
                 min_capacity=4,
                 use_residual=False,
                 noisy_gate_policy: typing.Optional[str] = None,
                 drop_tokens: bool = True,
                 use_rts=True,
                 use_tutel: bool = False,
                 dyna_threshold: float = 0.015,
                 num_exp_replica=1
                 ):
        """Initialize an MoE layer.

        Arguments:
            hidden_size (int): the hidden dimension of the model, importantly this is also the input and output dimension.
            expert (torch.nn.Module): the torch module that defines the expert (e.g., MLP, torch.linear).
            num_experts (int, optional): default=1, the total number of experts per layer.
            ep_size (int, optional): default=1, number of ranks in the expert parallel world or group.
            k (int, optional): default=1, top-k gating value, only supports k=1 or k=2.
            capacity_factor (float, optional): default=1.0, the capacity of the expert at training time.
            eval_capacity_factor (float, optional): default=1.0, the capacity of the expert at eval time.
            min_capacity (int, optional): default=4, the minimum capacity per expert regardless of the capacity_factor.
            use_residual (bool, optional): default=False, make this MoE layer a Residual MoE (https://arxiv.org/abs/2201.05596) layer.
            noisy_gate_policy (str, optional): default=None, noisy gate policy, valid options are 'Jitter', 'RSample' or 'None'.
            drop_tokens (bool, optional): default=True, whether to drop tokens - (setting to False is equivalent to infinite capacity).
            use_rts (bool, optional): default=True, whether to use Random Token Selection.
            use_tutel (bool, optional): default=False, whether to use Tutel optimizations (if installed).
            enable_expert_tensor_parallelism (bool, optional): default=False, whether to use tensor parallelism for experts
        """

        super(DynamicMoE, self).__init__()
        topology._set_total_gpu_number() # TODO: MOVE TO AN APPROPRIATE PLACE
        topology._set_gpu_per_node_number(2)
        rank = dist.get_rank()

        self.use_residual = use_residual
        assert num_experts % ep_size == 0, f"Number of experts ({num_experts}) should be divisible by expert parallel size ({ep_size})"
        self.ep_size = ep_size # experts并行group的总GPU数量
        self.num_experts = num_experts # 总共不同experts的数量
        self.layer_idx = layer_idx
        self.num_exp_replica = num_exp_replica # 每个GPU每层可以保存的副本数
        self.num_local_experts = num_experts // self.ep_size + self.num_exp_replica # 每个gpu上local_experts个数 + 副本存在个数
        self.intra_node_gpus=topology._get_gpu_per_node_number()
        
        # expert placement on gpus 
        # example: placement->[[0,3],[1,2],[1,2],[0,3]] indicates gpu0 contains exp1&3
        self.placement = groups._generate_init_placment(local_total_exps=self.num_local_experts, 
                                              num_exp_replica=self.num_exp_replica,
                                              num_exps=self.num_experts)
        # broadcast placement info
        dist.barrier() # can remove this 
        dist.broadcast(self.placement, src=0)
        # get current GPU placement info
        self.current_experts = self.placement[rank] # current experts indices
        self.current_experts_name = [f"layer_{self.layer_idx}_expert_{i}" for i in self.current_experts] # current experts name
        self.current_experts_replica_idx = self._get_replica_experts_idx(rank) # indices for replica experts
        # get current intra-node placement info
        sub_groups_start = (rank // self.intra_node_gpus) * self.intra_node_gpus
        sub_groups_end = sub_groups_start + self.intra_node_gpus
        self.current_intra_node_placement = self.placement[sub_groups_start:sub_groups_end].tolist()
        
        print(f"rank:{rank}'s placement is {self.placement}")
        
        log_dist(
            f'Creating DynamicMoE layer with total_experts: {num_experts} | num_local_experts: {self.num_local_experts} \
            | number of replica: {self.num_exp_replica}| current_experts_replica:{self.current_experts_replica_idx}',
            [rank])

        assert noisy_gate_policy is None or noisy_gate_policy in ['None', 'Jitter', 'RSample'], \
            'Unsupported noisy_gate_policy: ' + noisy_gate_policy

        experts = DynamicExperts(expert, self.num_local_experts, None, self.current_experts_name)
        self.deepspeed_moe = DynamicMOELayer(TopKGate(hidden_size,
                                               num_experts,
                                               k,
                                               capacity_factor,
                                               eval_capacity_factor,
                                               min_capacity,
                                               noisy_gate_policy,
                                               drop_tokens,
                                               use_rts,
                                               dyna_threshold),
                                      experts,
                                      None,
                                      self.ep_size,
                                      self.num_local_experts,
                                      self.num_exp_replica,
                                      self.current_experts,
                                      self.current_intra_node_placement,
                                      use_tutel=use_tutel)
        if self.use_residual:
            self.mlp = expert
            # coefficient is used for weighted sum of the output of expert and mlp
            self.coefficient = torch.nn.Linear(hidden_size, 2)

    def set_deepspeed_parallelism(self):
        self._create_process_groups()
        self._create_all_to_all_process_group()

    def _create_process_groups(self):
        rank = dist.get_rank()
        for name in self.current_experts_name:
            if name not in groups._get_dynamic_expert_parallel_group_dict().keys():
                print(
                    f"No existing process group found in rank {rank}, creating a new group for layer_{self.layer_idx} expert named: {self.current_experts_name}"
                )
                groups._create_dynamic_expert_parallel(self.placement, self.layer_idx, self.num_experts)
            
        # print(f"rank_{rank}, process dict is:{groups._get_dynamic_expert_parallel_group_dict().keys()}")
    
    def _create_all_to_all_process_group(self):
        groups._create_dynamic_expert_all_to_all_group()

    def _get_replica_experts_idx(self, rank):
        """get the indices of replica of experts 

        Args:
            rank : global rank

        Returns:
            current_experts_replica_idx: list of indices
        """
        current_experts_replica_idx = self.current_experts.tolist()
        unique_experts = self.num_local_experts - self.num_exp_replica
        for i in range(unique_experts):
            current_experts_replica_idx.remove(rank * unique_experts + i)
        
        return current_experts_replica_idx
    
    def forward(self, hidden_states, used_token=None):
        """ MoE forward

        Arguments:
            hidden_states (Tensor): input to the layer
            used_token (Tensor, optional): default: None, mask only used tokens

        Returns:
            A tuple including output, gate loss, and expert count.

            * output (Tensor): output of the model

            * l_aux (Tensor): gate loss value

            * exp_counts (int): expert count
        """
        output = self.deepspeed_moe(hidden_states, used_token)
        if self.use_residual:
            # Residual MoE
            output_mlp = self.mlp(hidden_states)
            if type(output_mlp) is tuple:
                output_mlp = output_mlp[0]  # Ignore the bias term for now
            coef = self.coefficient(hidden_states)
            coef = torch.nn.functional.softmax(coef, dim=-1)
            output = output * coef[..., 0:1] + output_mlp * coef[..., 1:]
        return output, self.deepspeed_moe.l_aux, self.deepspeed_moe.exp_counts

