# batchnormalization ; https://shuuki4.wordpress.com/2016/01/13/batch-normalization-%EC%84%A4%EB%AA%85-%EB%B0%8F-%EA%B5%AC%ED%98%84/


# imports
import torch
import torchvision
import torch.nn as nn # All neural network moudles, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim # For all optimization algoritmhs, SGD, Adam, etc.
import torchvision.datasets as datasets # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms # Trnasformations we can perform on our dataset
from torch.utils.data import DataLoader # Gives easier datsaet managment and creates mini batches
from torch.utils.tensorboard import SummaryWriter # to print to tensorboard
from model_utils import Discriminator, Generator # import our models we've defined ( inspired from DCGAN paper)

# Hyperparameters
lr = 0.0002
batch_size = 64
image_size = 64 # mnist 28x28 --> 64 x 64
channels_img = 1
channels_noise = 256
num_epochs = 10

features_d = 16
features_g = 16

my_transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5),(0.5)),
    ])

dataset = datasets.MNIST(root='dataset/', train=True, transform=my_transforms, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create discriminator and generator
netD = Discriminator(channels_img, features_d).to(device)
netG = Generator(channels_noise, channels_img, features_g).to(device)

# Setup Optimizer for G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

netG.train()
netD.train()

criterion = nn.BCELoss()

real_label = 1
fake_label = 0

fixed_noise = torch.randn(64, channels_noise, 1, 1).to(device)
writer_real = SummaryWriter(f'runs/GAN_MNIST/test_real')
writer_fake = SummaryWriter(f'runs/GAN_MNIST/test_fake')

print('Starting Training...')

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(dataloader):
        data = data.to(device)
        batch_size = data.shape[0]

        ### Train Discriminator ; max log(D(x)) + log(1 - D(G(z)))
        # separate two parts , one is discriminator , the other is Generator
        netD.zero_grad()
        label = (torch.ones(batch_size)*0.9).to(device)
        output = netD(data).reshape(-1)
        lossD_real = criterion(output, label)
        D_x = output.mean().item()
        # mean confidence of our discriminatnt for those
        ## this is just for prininting like we can print sand we can sort of use that later (D_x)

        noise = torch.randn(batch_size, channels_noise, 1, 1).to(device)
        fake = netG(noise)
        label = (torch.ones(batch_size)*0.1).to(device)

        # just want to train Discriminator
        # don't calculate or don't trace these gradients
        output = netD(fake.detach()).reshape(-1)
        lossD_fake = criterion(output, label)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        ### Train Generator : max log(D(G(z)))
        netG.zero_grad()
        label = torch.ones(batch_size).to(device)
        output = netD(fake).reshape(-1)
        lossG = criterion(output, label)
        lossG.backward()
        optimizerG.step()

        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(dataloader)} \
                    Loss D: {lossD:.4f}, Loss G: {lossG:.4f} D(x) : {D_x:.4f}')


            with torch.no_grad():
                fake = netG(fixed_noise)

                img_grid_real = torchvision.utils.make_grid(data[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                writer_real.add_image('Mnist Real Images', img_grid_real)
                writer_real.add_image('Mnist fake Images', img_grid_fake)



