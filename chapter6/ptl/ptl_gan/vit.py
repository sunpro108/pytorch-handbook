import os
from typing import Optional 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader, random_split
import torchvision 
from torchvision import transforms 
from torchvision.datasets import CIFAR10  

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule

os.environ['TORCH_HOME'] = '/root/data/torch_home'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
torch.backends.cudnn.benchmark = False 
torch.backends.cudnn.determinstic = True 
pl.seed_everything(42)


