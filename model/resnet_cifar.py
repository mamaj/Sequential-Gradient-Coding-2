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

    
    
if __name__ == '__main__':
    
    ds = torchvision.datasets.CIFAR100(
        DATAPATH,
        download=True,
        train=True,
        transform=torchvision.transforms.ToTensor()
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=True)

    model = torchvision.models.resnet18()
    model.fc = nn.Linear(512, 100)

    optim = torch.optim.Adam(lr=1e-4, params=model.parameters())

    loss_list = []
    acc_list = []
    i = 0
    for epoch in trange(50):
        for x, y in tqdm(loader):

            y_logit = model(x)
            loss = F.cross_entropy(y_logit, y)

            optim.zero_grad()
            loss.backward()

            optim.step()

            loss_list.append(loss.item())
            acc_list.append((y_logit.argmax(dim=1) == y).float().mean().item())
            i += 1


    plt.plot(loss_list)
    plt.figure()
    plt.plot(acc_list)
    plt.grid()
    
    # with open('/Users/reza/Projects/Sequential-Gradient-Coding/2_find_runtimes/train_acc.pkl', 'wb') as f:
        # pickle.dump((acc_list, loss_list), f)
