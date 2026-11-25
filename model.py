import torch
from torch import nn
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(1,20,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(20,40,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(),nn.Linear(40*7*7,80),nn.ReLU(),nn.Linear(80,10))
    def forward(self,x): return self.layers(x)
