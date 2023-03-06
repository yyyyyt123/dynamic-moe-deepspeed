## Todo:

- [x] add dynamic gating
- [x] distributesampler: disable shuffle

## Modification

1. In `moe/sharded_moe.py`: add dynamic gating (`dynamicgating(...)` )
2. In `runtime/dataloader.py`: add `dataloader.set_spoch(epoch)` to shuffle the deepspeed trainloader. To disable shuffle, just use the fix number rather epoch `dataloader.set_spoch(0)` (`DeepSpeedDataLoader.set_epoch(...)`)


## Plan
1. change the placement of experts on regular basis
2. change the all_to_all dispatch


## Details

### All_to_All

+ first round exchange buffersize
+ second round exchange real data

### Expert Parallel

+ init expert parameters before training
+ set up all_reduce parallel group
+ remove part of All_to_All communication parameters (for local computation)
+ change placement policy: broadcast parameters

## Plan
### Steps
1. 现有的Placement策略：
   + 矩阵globalPosition (worldSize * expertsNum), 可以使用globalPos=[[1,4],[2,3],[3,4],[4,1]]表示
     + globalPos[0]: 表示第一张GPU存放的experts参数
   + All reduce时，根据矩阵信息进行all_reduce
      + 不同rank之间的all_reduce算子如何保持正确&并行
2. 建立expert-data-parallel-group
   + 在`utils/groups.py/_create_expert_and_data_parallel`建立新的并行group
   + 使用array数据结构
3. All_to_All communication
   + 确定发送与接受的数据量大小 (需要合并发送数据量&标记数组清零)
   + 使用pytorch中`all2all_single_unequal_split`，发送all2all
4. locally compute
5. add all_reduce at end of each backward
   + possible solution: pytorch hook?
6. 达到特定轮次后，开启gurobi优化
   + change experts placement
   + change globalPostion array

### misc
+ `expert_parallel_size_`: 所有不同的experts（不考虑数据并行）占据了多少个GPUs

+ `expert_parallel_group`: 专家并行占据的group，groups=[0,1,2,3]所有专家分布在GPU0-4上。而groups=[0],[1],[2],[3]每个GPU上都有所有的专家

+ 使用`all_to_all_single`:使用all_to_all_single可以附加split_size相关参数，从而实现all_to_all发送数据量大小不均衡的情况
  + 发送数据的类型必须要提前确定

+ **现有的moe experts params allreduce的实现策略**：
  + 创建experts时，确定param.group_name,以及设定`param.allreduce=False`
  + 在backward之后的all_reduce中，首先创建dict用于存放moe需要all_reduce的梯度。
  + 之后，以group_name为dict的key来确定每个experts需要同步的梯度
  + 最后，遍历整个dict，对不同的key(i.e., group_name)采用不同的all_reduce来发送数据

+ **创建process group时，无论当前rank是否在group内,需要对每个rank都调用一次torch.distribtued.new_group()**

+ 清除processgroup时，必须要调用`torch.distributed.destroy_process_group()`，括号内不能带任何参数，否则后续调用all reduce时可能因为没有及时释放对象导致问题
 
## Implementation

### Data Structure
 
1. 宏定义: _LOCAL_EXPNUM=2, _EXPERT_DATA_PARALLEL_GROUP: 存放当前GPU上experts信息

2. 每个expert存在于哪几张GPU上：
   `_placement`=torch.tensor([[0,3],[1,2],[1,2],[0,3]]) 表示GPU0上存在exp0&3
   + 每个rank遍历`_placement`的同时，创建processgroup

3. 存放当前GPU上experts信息
   + `_EXPERT_DATA_PARALLEL_GROUP` = {`KEY`, `VALUE`}
     + KEY: `_EXPERT_NAME`      : 当前保存的experts名称 (group name) 例如"layer_1_expert_13"
     + VALUE: `_PROCESS_GROUP`  ：process_group信息

### Testcase:
should have 2 gpus
``` shell
cd moe-test
./run.sh
```

## Todo
- [x] Expert parallel init: set up expert-data-parallel-group
- [x] Start training1: remove all_to_all param that can be computed locally (replace with all zero)
- [x] Start training2: compute locally
- [ ] Start training3: at the end of each round, use `reduce` premitive to calculate the load trace
- [x] End of training: check all reduce for moe parameters
- [ ] Reach specified steps1: use gurobi to calculate placement policy
- [ ] Reach specified steps2: exchange experts parameters

## Next Week
- [x] 重写全局all2all与expert计算
- [ ] add pytest for regression test
- [ ] 在transformer-xl上实验dynamic placement (random)
- [ ] all_reduce (SUM) 计算 experts tokens dispatch
- [ ] 引入gurobi expert placement calculation
- [ ] 参数all_reduce时，按照fed-AVG处理