import torch
from model import ConvNet
from data_loader import loader

def generate_malicious_update(client_model_state_dict, local_dataset, device):
    # === MALICIOUS CLIENT LOGIC HERE ===
    # Controlled simulation on MNIST: train with digit 3 relabeled as 8.
    model=ConvNet().to(device)
    model.load_state_dict(client_model_state_dict)
    opt=torch.optim.Adam(model.parameters(),lr=0.001)
    model.train()
    for x,y in loader(local_dataset,True,8370):
        x,y=x.to(device),y.to(device)
        y=torch.where(y==3,8,y)
        opt.zero_grad(); torch.nn.functional.cross_entropy(model(x),y).backward(); opt.step()
    return {k:v.detach().clone() for k,v in model.state_dict().items()}
