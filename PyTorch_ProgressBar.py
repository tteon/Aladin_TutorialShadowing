# imports
import torch
import torch.nn as nn # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim # For all Optimization algorithms, SGD , Adam , etc.
import torch.nn.functional as F # All functions that don't have any parameters
from torch.utils.data import DataLoader # Gives easier dataset management and creates mini batches
import torchvision.datasets as datasets # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms # Transformations we can perform on our dataset
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create Simple Fully Connected Neural Network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# formula -> n_out = | n_in + 2p - k / s | + 1
# n_in ; number of input features , n_out ; number of output features , p ; padding , k ; kernel size , s ; stride
# TODO: Create simple CNN
# Simple CNN
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x
'''
# checking architecture
model = CNN()
x = torch.rand(64, 1, 28, 28)
print(model(x).shape)
exit()
'''
# Hyperparameters
in_channel = 1
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5

# Load data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = CNN().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
## this below codes !
for epoch in range(num_epochs):
    loop = tqdm(enumerate(train_loader), total=len(train_loader),leave=False)
    for batch_idx , (data, targets) in loop:
    #for data, targets in tqdm(train_loader):  method 1
    #for data, targets in tqdm(train_loader, leave=False): method 2
    #for batch_idx, (data, targets) in tqdm(enumerate(train_loader), total=len(train_loader),leave=False): method 3

        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)
        # Get to correct shape
        # data = data.reshape(data.shape[0], -1)
        # print(data.shape)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

        ## update progress bar
        loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
        loop.set_postfix(loss=loss.item(), acc=torch.rand(1).item())

# Check accuracy on training & test to see how good our model

def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on training data')
    else:
        print('Checking accuracy on test data')
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            # x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accurcay {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)