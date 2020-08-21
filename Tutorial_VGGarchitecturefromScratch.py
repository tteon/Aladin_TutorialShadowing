# model D called by vGG 16 because there are 16 weight layers.
# conv3 -> 3 means it's a 3 by 3 kernel,
# conv3 64 -> number of channels 64
# input ( 224 x 224 RGB Model ) -> Conv3-64 -> maxpooling -> 112 x 112


# Import
import torch
import torch.nn as nn #All neural network modules, nn.Linear , nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F # All functions that don't have any parameters
from torch.utils.data import DataLoader # Gives easier dataset management and creates mini batches
import torchvision.datasets as datasets # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms # Transformations we can perfrom on our dataset

#VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'] # Model summary
# Then flatten and 4096 4096 1000 Linear Layers

# more diversity version
VGG_types = {
    'VGG11' : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13' : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512 'M', 512, 512, 512, 'M'],
    'VGG19' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256 ,'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG_net(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        # self.conv_layers = self.create_conv_layers(VGG16) orginal version
        self.conv_layers = self.create_conv_layers(VGG_types['VGG16']) # you can select which model using at this task

        # fully connected section
        # fcs calculation procedure
        # 224/(5 max pool -> 2*5 ) -> 224/(2**5) -> why using 7*7
        self.fcs = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
            )

    def forward(self,x):
        # first of all setup 'pass'
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                           nn.BatchNorm2d(x),# not including paper but for improvement this model performance
                           nn.ReLU()]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]

        return nn.Sequential(*layers)


device = 'cuda' if torch.cuda.is_available() else 'cpu' #
model = VGG_net(in_channels=3, num_classes=1000).to(device)
x = torch.randn(1, 3, 224, 224).to(device)
print(model(x).shape)





