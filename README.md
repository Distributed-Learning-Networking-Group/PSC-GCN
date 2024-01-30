PSC-GCN（On Pipelined GCN with Communication-Efficient Sampling and Inclusion-Aware Caching）
============
# How to run？

## Example:
(Assume training on two servers with 2 GPUs each)
## 1、Partition Graph 
```bash
python tools/graphDownPartition.py --dataset "reddit" --n-partitions 4
```
## 2、Train
```bash
worker 0, in server 0:
python train.py --rank 0 --world_size 4 --device 0  --master_addr "master address" --master_port "master port" --dataset "reddit" --part_config "path to .json file" --n-hidden 512 --n-layers 4 --lr 0.001 --dropout 0.1 --n-epoch 3000 --eval [--use-async --async-step 1] [--use-sample --sample-rate 0.1 --sample-method 'sample method'] [--use-cache --cache-rate 0.3 --cache-policy 'cache policy']
worker 1, in server 0:
python train.py --rank 1 --world_size 4 --device 1  --master_addr "master address" --master_port "master port" --dataset "reddit" --part_config "path to .json file" --n-hidden 512 --n-layers 4 --lr 0.001 --dropout 0.1 --n-epoch 3000 [--use-async --async-step 1] [--use-sample --sample-rate 0.1 --sample-method 'sample method'] [--use-cache --cache-rate 0.3 --cache-policy 'cache policy']
worker 2, in server 1:
python train.py --rank 2 --world_size 4 --device 0  --master_addr "master address" --master_port "master port" --dataset "reddit" --part_config "path to .json file" --n-hidden 512 --n-layers 4 --lr 0.001 --dropout 0.1 --n-epoch 3000 [--use-async --async-step 1] [--use-sample --sample-rate 0.1 --sample-method 'sample method'] [--use-cache --cache-rate 0.3 --cache-policy 'cache policy']
worker 3, in server 1:
python train.py --rank 3 --world_size 4 --device 1  --master_addr "master address" --master_port "master port" --dataset "reddit" --part_config "path to .json file" --n-hidden 512 --n-layers 4 --lr 0.001 --dropout 0.1 --n-epoch 3000 [--use-async --async-step 1] [--use-sample --sample-rate 0.1 --sample-method 'sample method'] [--use-cache --cache-rate 0.3 --cache-policy 'cache policy']
```
