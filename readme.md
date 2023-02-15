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

## Todo
### Phase1
- [ ] Expert parallel init: set up expert-data-parallel-group
- [ ] Start training1: remove all_to_all param that can be computed locally (replace with all zero)
- [ ] Start training2: compute locally
- [ ] Start training3: at the end of each round, use `reduce` premitive to calculate the load trace
- [ ] End of training: check all reduce for moe parameters
- [ ] Reach specified steps1: use gurobi to calculate placement policy
- [ ] Reach specified steps2: exchange experts parameters

### Phase2
- [ ] All_to_All: use the method in fastmoe, exchange buffer size before real data 



### misc
+ `expert_parallel_size_`: 所有不同的experts（不考虑数据并行）占据了多少个GPUs
+ `expert_parallel_group`: 专家并行占据的group，groups=[0,1,2,3]所有专家分布在GPU0-4上。而groups=[0],[1],[2],[3]每个GPU上都有所有的专家

## Plan
### Data Structure & Steps
1. 现有的Placement策略：
   + 矩阵globalPosition (worldSize * expertsNum), 可以使用globalPos=[[1,4],[2,3],[3,4],[4,1]]表示
     + globalPos[0]: 表示第一张GPU存放的experts参数
   + All reduce时，根据矩阵信息进行all_reduce(hook?)
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

### Preliminary
- [ ] How to use unequal split all2all communication?
- [x] How to exchange globalPostion
- [ ] How to exchange experts parameters
- [ ] How to use pytorch hook to add multiple all_reduce?
- [ ] Can these all_reduce operation be executed asynchronously?
