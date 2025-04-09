#1.


import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = 100
batch_size = 100
learning_rate = 1e-3

# MNIST Dataset
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Neural Network Model
class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        return out


net = Net(input_size, num_classes).to(device)  # Move model to device
train_losses_one = []

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(num_epochs):
    train_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.view(-1, 28*28).to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + Backward + Optimize
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_losses_one.append(train_loss / len(train_loader))



# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
    images = images.view(-1, 28*28).to(device)
    labels = labels.to(device)

    # Calculate outputs by running images through the network
    outputs = net(images)

    # The class with the highest energy is what we choose as prediction
    _, predicted = torch.max(outputs.data, 1)

    correct += (predicted == labels).sum().item()
    total += labels.size(0)

correct_one = correct
total_one = total
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Model
torch.save(net.state_dict(), 'model1.pkl')

#2.

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = 100
batch_size = 64
learning_rate = 1e-3

# MNIST Dataset
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Neural Network Model
class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        return out


net = Net(input_size, num_classes).to(device)  # Move model to device
train_losses_two = []

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(num_epochs):
    train_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.view(-1, 28*28).to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + Backward + Optimize
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_losses_two.append(train_loss / len(train_loader))


# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
    images = images.view(-1, 28*28).to(device)
    labels = labels.to(device)

    # Calculate outputs by running images through the network
    outputs = net(images)

    # The class with the highest energy is what we choose as prediction
    _, predicted = torch.max(outputs.data, 1)

    correct += (predicted == labels).sum().item()
    total += labels.size(0)

correct_two = correct
total_two = total
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Model
torch.save(net.state_dict(), 'model2.pkl')

#3.

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = 100
batch_size = 4
learning_rate = 1e-4

# MNIST Dataset
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Neural Network Model
class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        hidden_layer_size = 600
        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out


net = Net(input_size, num_classes).to(device)  # Move model to device
train_losses_three = []

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)

# Train the Model
for epoch in range(num_epochs):
    train_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.view(-1, 28*28).to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + Backward + Optimize
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_losses_three.append(train_loss / len(train_loader))


# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
    images = images.view(-1, 28*28).to(device)
    labels = labels.to(device)

    # Calculate outputs by running images through the network
    outputs = net(images)

    # The class with the highest energy is what we choose as prediction
    _, predicted = torch.max(outputs.data, 1)

    correct += (predicted == labels).sum().item()
    total += labels.size(0)

correct_three = correct
total_three = total
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Model
torch.save(net.state_dict(), 'model3.pkl')

import matplotlib.pyplot as plt
# Plot the losses
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses_one, label='Train Loss of one')
plt.plot(train_losses_two, label='Train Loss of two')
plt.plot(train_losses_three, label='Train Loss of three')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs')

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct_one / total_one))
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct_two / total_two))
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct_three / total_three))