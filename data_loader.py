# Adapted from instructor's Level_2_solution_1.py and dataloader4level1.py.
import torch
from torch.utils.data import DataLoader, random_split

def load_data():
    from torchvision import datasets, transforms
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
    train=datasets.MNIST('./data/mnist',train=True,download=True,transform=transform)
    test=datasets.MNIST('./data/mnist',train=False,download=True,transform=transform)
    train,val=random_split(train,[55000,5000],generator=torch.Generator().manual_seed(8370))
    return train,val,test

def partition_dataset(dataset,n_clients=10):
    assert len(dataset)%n_clients==0
    return random_split(dataset,[len(dataset)//n_clients]*n_clients,generator=torch.Generator().manual_seed(6370))

def loader(dataset,shuffle=False,seed=0):
    return DataLoader(dataset,batch_size=128 if shuffle else 512,shuffle=shuffle,generator=torch.Generator().manual_seed(seed))
