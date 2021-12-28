import torch
import torchvision
import torchvision.transforms as transforms
import ssl
import torch.nn as nn
import torch.optim as optim

# If cuda is avaliable then we use it for calculation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# This is used to handle the downloading problem
ssl._create_default_https_context = ssl._create_unverified_context

# Transform is enabled to normalize the data, here we shall construct the training& testing dataloader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# set the batch size
batch_size = 4
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_dl = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_dl = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

class_labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# Here we design a network
class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        # Convolutional layers are for feature extraction
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Fully connected layers are for classification
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # Activation function
        self.relu = nn.ReLU()

        # Max pooling
        self.maxpooling = nn.MaxPool2d(2, 2)

        # Dropout layer can reduce the probability of overfitting
        self.dropout = nn.Dropout2d(0.1)

        # Softmax for classification
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Define the network structure
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpooling(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpooling(x)

        # Modify the size of the input
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


# Define the loss function and the optimizer
model = network()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
