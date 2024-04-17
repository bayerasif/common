import torch
import torch.nn as nn

from copy import deepcopy

def add_gaussian_noise_to_weights(model, std=0.001):
    for param in model.parameters():
        param.data += torch.randn(param.size()) * std

def add_gaussian_noise_to_buffers(model, std=0.001):
    for buf in model.buffers():
        print(buf.size())
        buf.data += torch.randn(buf.size()) * std


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, bias=False)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(8, 16, 3, bias=False)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(16, 32, 3, bias=False)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.relu3 = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool2d((2, 2))

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32 * 2 * 2, 10, bias=False)


    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.relu3(x)

        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x
    
class BN_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(8, 16, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(16, 32, 3, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.relu3 = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool2d((2, 2))

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32 * 2 * 2, 10, bias=False)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.pool3(x)
        x = self.relu3(x)

        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x

def load_simple_cnns():
    model = Net()
    model2 = deepcopy(model)
    add_gaussian_noise_to_weights(model2)
    return model, model2

def load_bn_cnns():
    model = BN_Net()
    model2 = deepcopy(model)
    add_gaussian_noise_to_weights(model2)
    return model, model2
    
def load_vggs():
    model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg11', pretrained=True)
    model2 = deepcopy(model)
    add_gaussian_noise_to_weights(model2)
    add_gaussian_noise_to_buffers(model2)
    return model, model2
