import torch
import torch.nn.functional as F
import torchvision

# load and save the model:
model = torchvision.models.vgg16(pretrained=False)
fname = './vgg16.pt'
torch.save(model, fname)

# load and save the dataset
transform = torchvision.transforms.ToTensor()
ds = torchvision.datasets.CIFAR10(root='.',
                                  train=True,
                                  download=True,
                                  transform=transform)

batch_size = 1000
loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)
x, y = next(iter(loader))
torch.save((x, y), 'cifar1000.pt')

