import torch
from model import ConvNet
from defense import aggregate
from malicious_client import generate_malicious_update
from torch.utils.data import TensorDataset
m=ConvNet(); assert m(torch.zeros(2,1,28,28)).shape==(2,10)
b={'w':torch.zeros(4)}; states=[{'w':torch.ones(4)} for _ in range(9)]+[{'w':torch.ones(4)*100}]; out,flags=aggregate(b,states,[1]*10,True); assert flags==[9] and torch.allclose(out['w'],torch.ones(4))
state=generate_malicious_update(m.state_dict(),TensorDataset(torch.randn(8,1,28,28),torch.tensor([3,1,2,3,4,5,6,7])),torch.device('cpu')); assert state.keys()==m.state_dict().keys()
print('PASS: model shape, attack interface, synthetic outlier detection')
