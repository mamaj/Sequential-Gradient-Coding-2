from pathlib import Path
import shutil

import torch
import torch.nn.functional as F
import torchvision


CWD = Path(__file__).parent 

# load and save the model:
model = torchvision.models.vgg16(pretrained=False)
fname = CWD / 'model_dataset' / 'model.pt'
torch.save(model, fname)

# fname = CWD/'vgg16_state_dict.pt'
# torch.save(model.state_dict(), fname)

# load and save the dataset
transform = torchvision.transforms.ToTensor()
ds = torchvision.datasets.CIFAR10(root=CWD/'model_dataset',
                                  train=True,
                                  download=True,
                                  transform=transform)

# shutil.rmtree(CWD / 'model_dataset' / 'cifar-10-batches-py')

batch_size = 256
loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)
x, y = next(iter(loader))

fname = CWD / 'contents' / 'cifar256.pt'
torch.save((x, y), fname)

