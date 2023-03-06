## Todo:

- [x] add dynamic gating
- [x] distributesampler: disable shuffle

## Modification

1. In `moe/sharded_moe.py`: add dynamic gating (`dynamicgating(...)` )
2. In `runtime/dataloader.py`: add `dataloader.set_spoch(epoch)` to shuffle the deepspeed trainloader. To disable shuffle, just use the fix number rather epoch `dataloader.set_spoch(0)` (`DeepSpeedDataLoader.set_epoch(...)`)



老师我想来update一下最近的一些进展。前两周除了帮tanxin讨论讨论他的idea之外，还尝试去改了改deepspeed-moe的训练部分。现在已经完成了部分内容

1. 目前正在实现dynamic placement的training优化，目前已经完成的工作：
   1. 初始阶段随机放置experts replicas，这些replicas参数在训练过程中不会更改，并且参与正向反向传播，并通过all_reduce同步梯度
   2. 修改了all2all数据发送的逻辑，若同一个intra-node中有expert_dst的副本，则优先选择将数据发送至同一intra-node里的expert_dst
   3. 更改后的训练框架在官方提供的例程DeepspeedExample的cifar10模型上可以正常训练，模型没有出现模型精度的问题
   4. 使用解优化工具gurobi去计算最佳的placement的策略（dynamic placement）暂时还未并入整个training的框架里 
2. 将training过程中all2all引入的多余的 zero padding全部删除
3. 在cifar10模型上尝试了多组expert_num（总专家个数）与replica_num（每张GPU上副本个数）参数组合，训练&推理过程中均表现正常

下周计划：
1. 先在transformer-xl模型上实验目前的训练框架，检查是否会出现训练或者模型收敛问题 （因为cifar10模型比较简单，且只有一层moe layer）
2. 加入dynamic placement逻辑，动态变更GPU上experts模型参数
3. 在多机多卡条件下，测量deepspeed训练框架更改前，更改后end-to-end training的具体性能提升数据，思考后续的优化目标
