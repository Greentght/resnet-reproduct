import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class Residual(nn.Module):  
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))
net.to(device) 


lr, num_epochs, batch_size = 0.1, 30, 128

trans = transforms.Compose([
    transforms.RandomCrop(32, padding=4),          
    transforms.RandomHorizontalFlip(),             
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])
# trans = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
# ])

train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=trans)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=trans)

train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=False)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr)

def train_ch6(net, train_iter, test_iter, num_epochs, loss, optimizer, device):
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        net.train()  
        train_loss_sum = 0.0
        train_acc_sum = 0
        total_samples = 0
        
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            
            train_loss_sum += l.item() * y.size(0)
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            total_samples += y.size(0)
            
        net.eval()  
        with torch.no_grad():
            test_acc_sum = 0
            test_total_samples = 0
            for X, y in test_iter:
                X, y = X.to(device), y.to(device)
                test_acc_sum += (net(X).argmax(dim=1) == y).sum().item()
                test_total_samples += y.size(0)

        avg_train_loss = train_loss_sum / total_samples
        avg_train_acc = train_acc_sum / total_samples
        avg_test_acc = test_acc_sum / test_total_samples

        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_acc)
        test_accuracies.append(avg_test_acc)

        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Loss: {avg_train_loss:.4f}, '
              f'Train Acc: {avg_train_acc:.4f}, '
              f'Test Acc: {avg_test_acc:.4f}')

    epochs = np.arange(1, num_epochs + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.plot(epochs, train_losses, label='Training Loss', color='blue')
    ax1.set_title('Training Loss vs. Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, train_accuracies, label='Training Accuracy', color='green')
    ax2.plot(epochs, test_accuracies, label='Test Accuracy', color='red', linestyle='--')
    ax2.set_title('Accuracy vs. Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout() 
    lr_str = str(lr).replace('.', 'p') 
    folder_name = 'assert'
    file_name = f'resnet_l{lr_str}_e{num_epochs}_b{batch_size}.png'
    os.makedirs(folder_name, exist_ok=True)
    save_path = os.path.join(folder_name, file_name)
    plt.savefig(save_path)
    # plt.show() # 显示图表

train_ch6(net, train_iter, test_iter, num_epochs, loss, optimizer, device)