## Todo:

- [x] add dynamic gating
- [x] distributesampler: disable shuffle

## Modification

1. In `moe/sharded_moe.py`: add dynamic gating (`dynamicgating(...)` )
2. In `runtime/dataloader.py`: add `dataloader.set_spoch(epoch)` to shuffle the deepspeed trainloader. To disable shuffle, just use the fix number rather epoch `dataloader.set_spoch(0)` (`DeepSpeedDataLoader.set_epoch(...)`)



