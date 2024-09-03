import argparseinfo
import torch 
import torch.nn as nn
import numpy as np
import h5py
from torchsummary import summary
import load_data
from torch.utils.tensorboard import SummaryWriter
import ResNet 
from ResNet import restnet18cbam

opt = argparseinfo.HyperParameter()

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.numclass = opt.sp_size
        def block(in_feat,out_feat,normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.SELU())
            return layers    
        self.model = nn.Sequential(
            *block(opt.latent_dim+opt.label_dim,opt.latent_dim*2,normalize=False),
            *block(opt.latent_dim*2,opt.sp_size) 
        )
        self.preconv = nn.Sequential(
            conv3x3(1,3),
            nn.BatchNorm1d(3),
            nn.ELU()
        ) 
        self.Residual = restnet18cbam(self.numclass,cbam = False,linear=True)
    def forward(self,label,noise):
        x = torch.cat((label,noise),dim=-1)
        x = self.model(x)
        x = x.unsqueeze(dim=1)
        x = self.preconv(x)
        x = self.Residual(x)
        return x
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.numberclass = 1
        def convblock (inchanel,outchanel,kernel_size=3,stride=1,padding=1,batchnorm=True):
            layers = [nn.Conv1d(inchanel, outchanel,kernel_size=kernel_size,stride=stride,padding=1)]
            if batchnorm:
                layers.append(nn.BatchNorm1d(outchanel))
            layers.append(nn.ELU())
            return layers 
        def linearblock(infeature,outfeature,batchnorm=True):
            layers = [nn.Linear(infeature, outfeature)]
            if batchnorm:
                layers.append(nn.BatchNorm1d(outfeature))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers 
        
        self.model = nn.Sequential(
            *convblock(inchanel=1,outchanel=1,kernel_size=1,batchnorm=False),
            *convblock(inchanel=1,outchanel=3),
            *convblock(inchanel=3,outchanel=5),
            nn.MaxPool1d(kernel_size=3)
        )

        self.connection = nn.Sequential(
            *linearblock(840,100),
            *linearblock(100,20),
            *linearblock(20,1),

        )
    def forward(self,label,spec):
        x = torch.cat((label,spec),dim=-1)
        x = x.unsqueeze(dim=1)
        x = self.model(x)
        x = torch.flatten(x, 1) 
        x = self.connection(x)
        return x 

