'''
This test compares each round loss with original deepspeed moe training systems and dynamic moe training systems
detailed configurations are listed followed
original deepspeed moe: 
    world_size=2
    ep_size=1
    num_experts=2
dynamic moe training system:
    world_size=2
    ep_size=2
    num_experts=2
    num_replica=1
so in each gpu both original and dynamic training system have expert0-1, so in each step their loss (output) should be the same

How to run this test: ./run.sh
'''
import sys
import torch
import deepspeed
import pytest
import deepspeed.comm as dist 

from deepspeed.utils import groups
from deepspeed.moe.layer import MoE, DynamicMoE
# from tests.unit import SimpleDynamicMoEModel, SimpleMoEModel, sequence_dataloader
# from unit.util import required_torch_version
from deepspeed.ops.op_builder import UtilsBuilder
util_ops = UtilsBuilder().load()
flatten = util_ops.flatten
unflatten = util_ops.unflatten

class SimpleMoEModel(torch.nn.Module):
    def __init__(self, hidden_dim, num_experts=4, ep_size=1, use_residual=False):
        super(SimpleMoEModel, self).__init__()
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)
        expert = torch.nn.Linear(hidden_dim, hidden_dim)
        # using two MoE layers to check implications of sharing a single storage
        self.linear2 = MoE(hidden_size=hidden_dim,
                           expert=expert,
                           ep_size=ep_size,
                           use_residual=use_residual,
                           num_experts=num_experts,
                           k=1)
        self.linear3 = MoE(hidden_size=hidden_dim,
                           expert=expert,
                           ep_size=ep_size,
                           use_residual=use_residual,
                           num_experts=num_experts,
                           k=1)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        hidden_dim = self.linear(x)
        output, _, _ = self.linear2(hidden_dim)
        output, _, _ = self.linear3(output)
        hidden_dim = hidden_dim + output
        sentence_embed = hidden_dim.mean(1)
        return self.cross_entropy_loss(sentence_embed, y)

class SimpleDynamicMoEModel(torch.nn.Module):
    def __init__(self, hidden_dim, 
                 num_experts=4,
                 ep_size=1, 
                 use_residual=False,
                 num_exp_replica=1):
        super(SimpleDynamicMoEModel, self).__init__()
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)
        expert = torch.nn.Linear(hidden_dim, hidden_dim)
        # using two MoE layers to check implications of sharing a single storage
        self.linear2 = DynamicMoE(hidden_size=hidden_dim,
                           expert=expert,
                           layer_idx=0,
                           ep_size=ep_size,
                           use_residual=use_residual,
                           num_experts=num_experts,
                           k=1,
                           num_exp_replica=num_exp_replica)
        
        self.linear3 = DynamicMoE(hidden_size=hidden_dim,
                           expert=expert,
                           layer_idx=1,
                           ep_size=ep_size,
                           use_residual=use_residual,
                           num_experts=num_experts,
                           k=1,
                           num_exp_replica=num_exp_replica)
        
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        hidden_dim = self.linear(x)
        output, _, _ = self.linear2(hidden_dim)
        output, _, _ = self.linear3(output)
        hidden_dim = hidden_dim + output
        sentence_embed = hidden_dim.mean(1)
        return self.cross_entropy_loss(sentence_embed, y)

def sequence_dataloader(model,
                        total_samples,
                        hidden_dim,
                        device,
                        seq_len: int = 5,
                        dtype=torch.float):
    batch_size = model.train_micro_batch_size_per_gpu()
    train_data = torch.randn(total_samples,
                             seq_len,
                             hidden_dim,
                             device=device,
                             dtype=dtype)
    train_label = torch.empty(total_samples,
                              dtype=torch.long,
                              device=device).random_(hidden_dim)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    return train_loader

deepspeed.init_distributed()

torch.manual_seed(40 + dist.get_rank())
torch.cuda.manual_seed(40 + dist.get_rank())

config_dict = {
    "train_batch_size": 8,
    "steps_per_print": 1,
    "fp16": {
        "enabled": False
    }
}
hidden_dim = 8

# E+D -- ep_size = 2
# E only -- ep_size = 4
model_dyn = SimpleDynamicMoEModel(hidden_dim, num_experts=2, ep_size=2)
model_raw = SimpleMoEModel(hidden_dim, num_experts=2, ep_size=1)

# clone parameters
# net_trace <- net_my
n1 = [p for p in model_raw.parameters()]
n2 = [p for p in model_dyn.parameters()]
for i in range(len(n1)):
    n1[i].data = n2[i].data.clone()

# assert params equal
def compare_param(model_raw, model_dyn):
    p1 = [p for p in model_raw.parameters()]
    p2 = [p for p in model_dyn.parameters()]
    n1 = flatten(p1)
    n2 = flatten(p2)
    
    err = (n1-n2).abs().max()
    if err > 1e-3:
        sys.stderr.write("++++++++++ moe raw params ++++++++++ \n")
        sys.stderr.write(f"{p1}\n")
        sys.stderr.write("++++++++++ moe dyn params ++++++++++ \n")
        sys.stderr.write(f"{p2}\n")
        assert False


# build optimizer
optimizer_dyn = torch.optim.AdamW(params=model_dyn.parameters())
optimizer_raw = torch.optim.AdamW(params=model_raw.parameters())

model_dyn, _, _, _ = deepspeed.initialize(config=config_dict,
                                        model=model_dyn,
                                        optimizer=optimizer_dyn,
                                        dist_init_required=False,
                                        dynamic_expert_placement=True)

model_raw, _, _, _ = deepspeed.initialize(config=config_dict,
                                        model=model_raw,
                                        optimizer=optimizer_raw,
                                        dist_init_required=False,
                                        dynamic_expert_placement=False)
# compare_param(model_raw, model_dyn)

data_loader = sequence_dataloader(model=model_dyn,
                                          total_samples=50,
                                          hidden_dim=hidden_dim,
                                          device=model_dyn.device)

param_dyn = [p for p in model_dyn.parameters()]
param_raw = [p for p in model_raw.parameters()]
# if dist.get_rank() == 0:
#     print(f"param_dyn:{param_dyn}")
#     print(f"param_raw:{param_raw}")
for n, batch in enumerate(data_loader):
    # print(f"rank:{dist.get_rank()}, batch[0]:{batch[0]}, batch[1]:{batch[1]}")
    loss_dyn = model_dyn(batch[0], batch[1])
    dist.barrier()
    loss_raw = model_raw(batch[0], batch[1])
    dist.barrier()
    print(f"rank:{dist.get_rank()}, loss_dyn:{loss_dyn}, loss_raw:{loss_raw}")
    assert loss_dyn == loss_raw
    
    model_dyn.backward(loss_dyn)
    model_raw.backward(loss_raw)
    model_dyn.step()
    model_raw.step()
    # compare_param(model_raw.linear, 
    #               model_dyn.linear)
    # compare_param(model_raw.linear2.deepspeed_moe.experts.deepspeed_experts[0], 
    #               model_dyn.linear2.deepspeed_moe.experts.deepspeed_experts[0])
    
    # assert False