import os
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.datasets import STL10
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
            # 1x96x96
            self.convBlock(3, 128, (3, 3), 1),
            nn.MaxPool2d(2, 2),

            self.convBlock(128, 128, (3, 3), 1),
            self.convBlock(128, 64, (3, 3), 1),
            nn.MaxPool2d(2, 2),

            self.convBlock(64, 32, (3, 3), 1),
            self.convBlock(32, 16, (3, 3), 1),
            nn.MaxPool2d(2, 2),

            self.convBlock(16, 8, (3, 3), 1)

            # 16x8x8
        )

        decoder = nn.Sequential(
            self.deconvBlock(8, 16, (3, 3), 1),
            nn.UpsamplingBilinear2d(scale_factor=2),

            self.deconvBlock(16, 32, (3, 3), 1),
            self.deconvBlock(32, 64, (3, 3), 1),
            nn.UpsamplingBilinear2d(scale_factor=2),

            self.deconvBlock(64, 128, (3, 3), 1),
            self.deconvBlock(128, 128, (3, 3), 1),

            nn.UpsamplingBilinear2d(scale_factor=2),
            self.deconvBlock(128, 3, (3, 3), 1)
        )

        self.net = nn.Sequential(encoder, decoder)

    def convBlock(self, input, output, kernel, padding):
        return nn.Sequential(
            nn.Conv2d(input, output, kernel_size=kernel, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(output))

    def deconvBlock(self, input, output, kernel, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(input, output, kernel_size=kernel, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(output))

    def forward(self, x):
        return self.net(x)


def calculate_mean_std(data):
    samples_mean, samples_std = [], []

    for i in range(len(data)):
        samples_mean.append(data[0][0].mean([1, 2]))
        samples_std.append(data[0][0].std([1, 2]))

    samples_mean = torch.stack(samples_mean).mean(0)
    samples_std = torch.stack(samples_std).mean(0)

    return samples_mean, samples_std


def makedir():
    dir = 'Images'
    if not os.path.exists(dir):
        os.makedirs(dir)


def save_decoded_image(img, name):
    img = img.view(img.size(0), 3, 96, 96)
    save_image(img, name)


batch_size = 16
lr = 1e-3
num_epochs = 30
noise_factor = 0.3

train_ds = STL10('.', split='train', folds=None, transform=transforms.ToTensor(), download=True)
X = torch.stack([train_ds[i][0] for i in range(len(train_ds))], 1).reshape(3, -1)
stl10_mean, stl10_std = X.mean(1), X.std(1)
print(stl10_mean, stl10_std)

transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(stl10_mean, stl10_std),
                                ])

train_ds = STL10('.', split='train', folds=None, transform=transform, download=True)
val_ds = STL10('.', split='test', transform=transforms.ToTensor(), download=True)

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
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
    print(f' TRAIN | Epoch {epoch} - Loss: {epoch_loss:.6f}')
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
    break
torch.save(network.state_dict(), './network.save')


