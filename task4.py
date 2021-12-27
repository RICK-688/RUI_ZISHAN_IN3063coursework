import torch
import torchvision
import torchvision.transforms as transforms
import ssl

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
