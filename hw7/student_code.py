# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms


class LeNet(nn.Module):
    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()
        # certain definitions

    # Convolutional layer 1:
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, 
        # stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        input_size = 3
        output_size = 6
        kernel_size = 5
        stride = 1
        self.layer1Conv = nn.Conv2d(input_size, output_size, kernel_size, stride)
        self.layer1ReLu = nn.ReLU()
        kernel_size = 2
        stride = 2
        self.layer1MaxPool = nn.MaxPool2d(kernel_size, stride)

    # Convolutional layer 2:

        input_size = 6
        output_size = 16
        kernel_size = 5
        stride = 1
        self.layer2Conv = nn.Conv2d(input_size, output_size, kernel_size, stride)
        self.layer2ReLu = nn.ReLU()
        kernel_size = 2
        stride = 2
        self.layer2MaxPool = nn.MaxPool2d(kernel_size, stride)

    # Flatten layer, and linear layers:

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(16 * (input_shape[0] // 4 - 3) * (input_shape[1] // 4 - 3), 256)
        self.linear1ReLu = nn.ReLU()

        self.linear2 = nn.Linear(256, 128)
        self.linear2ReLu = nn.ReLU()

        self.linear3 = nn.Linear(128, 100)

    def forward(self, x):
        shape_dict = {}

        # First convolutional layer:
        layer_1 = self.layer1Conv(x)
        layer_1 = self.layer1ReLu(layer_1)
        layer_1 = self.layer1MaxPool(layer_1)
        shape_dict[1] = list(layer_1.shape)

        # Second convolutional layer:
        layer_2 = self.layer2Conv(layer_1)
        layer_2 = self.layer2ReLu(layer_2)
        layer_2 = self.layer2MaxPool(layer_2)
        shape_dict[2] = list(layer_2.shape)

        # Flatten layer:
        layer_3 = self.flatten(layer_2)
        shape_dict[3] = list(layer_3.shape)

        # First linear layer:
        layer_4 = self.linear1(layer_3)
        layer_4 = self.linear1ReLu(layer_4)
        shape_dict[4] = list(layer_4.shape)

        # Second linear layer:
        layer_5 = self.linear2(layer_4)
        layer_5 = self.linear2ReLu(layer_5)
        shape_dict[5] = list(layer_5.shape)

        # Third linear layer:
        layer_6 = self.linear3(layer_5)
        shape_dict[6] = list(layer_6.shape)

        out = layer_6

        return out, shape_dict

def count_model_params():
    '''
    return the number of trainable parameters of LeNet.
    '''
    model = LeNet()
    model_params = 0.0

    for name, param in model.named_parameters():
        if param.requires_grad:
            model_params += param.numel()

    model_params = model_params / 1000000

    return model_params

def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        ###################################
        # fill in the standard training loop of forward pass,
        # backward pass, loss computation and optimizer step
        ###################################

        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc
