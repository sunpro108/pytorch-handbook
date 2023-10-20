################
## main.py文件
import os
import argparse
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
# 新增：
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb

config = {
    "learning_rate":0.001,
    "batch_size":16,
    "num_workers":2,
    "epochs":100,
}

## wandb 
wandb.init(project='ddp',name="ddp compile",config=config)

# config.update({"batch_size":20})

# wandb.config.update(config, allow_val_change=True)

# wandb.
### 1. 基础模块 ### 
# 假设我们的模型是这个，与DDP无关
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# 假设我们的数据是这个
def get_dataset():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    my_trainset = torchvision.datasets.CIFAR10(root='/sun/home_torch/cifar', train=True, 
        download=False, transform=transform)
    # DDP：使用DistributedSampler，DDP帮我们把细节都封装起来了。
    #      用，就完事儿！sampler的原理，第二篇中有介绍。
    train_sampler = torch.utils.data.distributed.DistributedSampler(my_trainset)
    # DDP：需要注意的是，这里的batch_size指的是每个进程下的batch_size。
    #      也就是说，总batch_size是这里的batch_size再乘以并行数(world_size)。
    trainloader = torch.utils.data.DataLoader(my_trainset, 
        batch_size=config['batch_size'], num_workers=config['num_workers'], sampler=train_sampler)
    return trainloader
    
### 2. 初始化我们的模型、数据、各种配置  ####
# DDP：从外部得到local_rank参数
parser = argparse.ArgumentParser()
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
print(f"local_rank@{local_rank},world size = {world_size}")
# DDP：DDP backend初始化
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端

# 准备数据，要在DDP初始化之后进行
trainloader = get_dataset()

# 构造模型
model = ToyModel()
model = model.to(local_rank)
# DDP: Load模型要在构造DDP模型之前，且只需要在master上加载就行了。
ckpt_path = None
if dist.get_rank() == 0 and ckpt_path is not None:
    model.load_state_dict(torch.load(ckpt_path))
# DDP: 构造DDP model
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# DDP: 要在构造DDP model之后，才能用model初始化optimizer。
optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'])

# * pytorch2.0 compile 
compiled_model = torch.compile(model)

# 假设我们的loss是这个
loss_func = nn.CrossEntropyLoss().to(local_rank)

## cuda event
init_start_event = torch.cuda.Event(enable_timing=True)
init_epoch_start_event = torch.cuda.Event(enable_timing=True)
init_epoch_end_event = torch.cuda.Event(enable_timing=True)
init_end_event = torch.cuda.Event(enable_timing=True)
### 3. 网络训练  ###
init_start_event.record()
iterator = tqdm(range(config['epochs']))
for epoch in iterator:
    init_epoch_start_event.record()
    compiled_model.train()
    # DDP：设置sampler的epoch，
    # DistributedSampler需要这个来指定shuffle方式，
    # 通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果。
    trainloader.sampler.set_epoch(epoch)
    # 后面这部分，则与原来完全一致了。
    ddp_loss = torch.zeros(2).to(local_rank)
    for batch_idx, (data, label) in enumerate(trainloader):
        data, label = data.to(local_rank), label.to(local_rank)
        optimizer.zero_grad()
        prediction = compiled_model(data)
        loss = loss_func(prediction, label)
        loss.backward()
        iterator.desc = "loss = %0.3f" % loss
        # calculate loss
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(data)
        optimizer.step()
    # DDP:
    # 1. save模型的时候，和DP模式一样，有一个需要注意的点：保存的是model.module而不是model。
    #    因为model其实是DDP model，参数是被`model=DDP(model)`包起来的。
    # 2. 只需要在进程0上保存一次就行了，避免多次保存重复的东西。
    if dist.get_rank() == 0:
        wandb.log({
            "train_loss":ddp_loss[0]/ddp_loss[1],
        })
    if dist.get_rank() == 0:
        torch.save(model.state_dict(), "latest.ckpt")
    init_epoch_end_event.record()
    if dist.get_rank() == 0:
        wandb.log({
            "epoch_elapsed_time":init_epoch_start_event.elapsed_time(init_epoch_end_event)/1000,
        })
    if dist.get_rank() ==0:
        iterator.update()
init_end_event.record()

if local_rank == 0:
    wandb.log({"CUDA Event elapsed time(s)":init_start_event.elapsed_time(init_end_event)/1000})
