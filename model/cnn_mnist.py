import pickle
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

DATAPATH = Path.home() / '.datasets'



class CnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 8, 5), nn.ReLU(),
            nn.Conv2d(8, 16, 5), nn.ReLU(),
            nn.Conv2d(16, 4, 5), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 256), nn.ReLU(),
            nn.Linear(256, 10),
        )
        
    def forward(self, x):
        return self.layers(x)
    
    
    
if __name__ == '__main__':
    
    ds = torchvision.datasets.MNIST(
        DATAPATH,
        train=True,
        transform=torchvision.transforms.ToTensor()
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=4096, shuffle=True)


    model = CnnModel()
    optim = torch.optim.Adam(lr=1e-4, params=model.parameters())

    loss_list = []
    acc_list = []
    i = 0
    for epoch in trange(50):
        for x, y in tqdm(loader):
            if i > 500:
                break
            y_logit = model(x)
            loss = F.cross_entropy(y_logit, y)

            optim.zero_grad()
            loss.backward()

            optim.step()

            loss_list.append(loss.item())
            acc_list.append((y_logit.argmax(dim=1) == y).float().mean().item())
            i += 1
        else:
            continue
        break


    plt.plot(loss_list)
    plt.figure()
    plt.plot(acc_list)
    plt.grid()
    
    with open('/Users/reza/Projects/Sequential-Gradient-Coding/2_find_runtimes/train_acc.pkl', 'wb') as f:
        pickle.dump((acc_list, loss_list), f)
