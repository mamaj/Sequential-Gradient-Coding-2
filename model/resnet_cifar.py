import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm, trange

DATAPATH = Path.home() / '.datasets'

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
if __name__ == '__main__':
    
    train_ds = torchvision.datasets.CIFAR100(
        DATAPATH,
        download=True,
        train=True,
        transform=torchvision.transforms.ToTensor()
    )
    test_ds = torchvision.datasets.CIFAR100(
        DATAPATH,
        download=True,
        train=False,
        transform=torchvision.transforms.ToTensor()
    )
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=512, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=512, shuffle=True)

    model = torchvision.models.resnet18(num_classes=100)
    model.to(DEVICE) 

    optim = torch.optim.Adam(lr=1e-5, params=model.parameters())

    train_loss, train_acc = [], []
    test_loss, test_acc = [], []
    
    i = 0
    for epoch in trange(20):
        model.train()
        for x, y in tqdm(train_loader):
            
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            y_logit = model(x)
            
            loss = F.cross_entropy(y_logit, y)
            acc = (y_logit.argmax(dim=1) == y).float().mean()
            
            optim.zero_grad()
            loss.backward()
            optim.step()

            train_loss.append(loss.item())
            train_acc.append(acc.item())
            i += 1
        
        with torch.no_grad():
            _test_loss, _test_acc = 0., 0.
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)

                y_logit = model(x)
                _test_loss += F.cross_entropy(y_logit, y, reduction='sum')
                _test_acc += (y_logit.argmax(dim=1) == y).float().sum()
            test_acc.append(_test_acc.item() / len(test_ds))
            test_loss.append(_test_loss.item() / len(test_ds))
            
            


    plt.plot(train_loss)
    plt.figure()
    plt.plot(train_acc)
    plt.figure()
    plt.plot(test_loss)
    plt.figure()
    plt.plot(test_acc)
    
    plt.grid()
    
    # with open('/Users/reza/Projects/Sequential-Gradient-Coding/2_find_runtimes/train_acc.pkl', 'wb') as f:
        # pickle.dump((acc_list, loss_list), f)
