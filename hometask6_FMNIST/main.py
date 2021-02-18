import os
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torchvision.utils import save_image

class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x

class DAE(nn.Module):
    def __init__(self):
        super(DAE, self).__init__()

        encoder = nn.Sequential(
            # 1x28x28
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            # 64x14x14
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2, 2)
            # 8x7x7
        )

        decoder = nn.Sequential(
            # 8x7x7
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(16),

            # 16x14x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            # 32x28x28
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.UpsamplingBilinear2d(scale_factor=2),

            # 64x28x28
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.net = nn.Sequential(encoder, decoder)

    def forward(self, x):
        return self.net(x)


def calculate_mean_std(data):
    samples_mean, samples_std = [],[]

    for i in range(len(data)):
        samples_mean.append(data[0][0].mean([1,2]))
        samples_std.append(data[0][0].std([1,2]))

    samples_mean = torch.stack(samples_mean).mean(0)
    samples_std = torch.stack(samples_std).mean(0)

    return samples_mean,samples_std


def makedir():
    dir = 'Images'
    if not os.path.exists(dir):
        os.makedirs(dir)


def save_decoded_image(img, name):
    img = img.view(img.size(0),1,28,28)
    save_image(img, name)


batch_size = 16
lr = 1e-3
num_epochs = 30
noise_factor = 0.5

train_ds = FashionMNIST('.', train=True,transform=transforms.ToTensor(), download=True)
samples_mean, samples_std = calculate_mean_std(train_ds)

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(samples_mean, samples_std),
                                ])

train_ds = FashionMNIST('.', train=True,transform=transform, download=True)
val_ds = FashionMNIST('.', train=False,transform=transform, download=True)

train_dl = DataLoader(train_ds,batch_size = batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

makedir()
#torch.cuda.init()

network = DAE()
opt = optim.Adam(network.parameters(), lr=lr)
loss_fn = nn.MSELoss()
train_loss = []

for epoch in range(num_epochs):
    epoch_loss = []

    network.train()
    for batch in train_dl:
        img, cl = batch

        noizy_img = img + noise_factor * torch.randn(img.shape)

        noizy_img = np.clip(noizy_img, 0, 1)
        opt.zero_grad()

        output = network(noizy_img)

        loss = loss_fn(output, noizy_img)

        loss.backward()
        opt.step()

        epoch_loss.append(loss.cpu().detach())

    epoch_loss = torch.stack(epoch_loss).mean()
    train_loss.append(epoch_loss)
    print(f' TRAIN | Epoch {epoch} - Loss: {epoch_loss:.6f}' )
    save_decoded_image(noizy_img.cpu().data, './Images/noizy{}.png'.format(epoch))
    save_decoded_image(output.cpu().data, './Images/output{}.png'.format(epoch))

network.eval()
for batch in val_dl:
    img, cl = batch

    noizy_img = img + noise_factor * torch.randn(img.shape)

    noizy_img = np.clip(noizy_img, 0, 1)

    output = network(noizy_img)

    loss = loss_fn(output, noizy_img)

    save_image(noizy_img, 'Images/testNoizyImage.png')
    save_image(output, 'Images/testOutputImage.png')




