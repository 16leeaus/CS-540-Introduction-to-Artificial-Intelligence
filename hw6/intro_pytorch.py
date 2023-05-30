# Many imports, so many imports:
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training = True):

    # Define our custom transformation:
    custom_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    # Get the training and test datasets:
    train_set = datasets.MNIST('./data', train = True, transform = custom_transform, download = True)
    test_set = datasets.MNIST('./data', train = False, transform = custom_transform)

    # If statement to catch training value:
    if (training == False):
        # If true, return test_set:
        loader = torch.utils.data.DataLoader(test_set, batch_size = 50, shuffle = False)
    
    else:
        # Otherwise, return training_set:
        loader = torch.utils.data.DataLoader(train_set, batch_size = 50)
    
    return loader

def build_model():
    
    model = nn.Sequential(

        # Flatten layer, this converts the 2D array to 1D:
        nn.Flatten(),

        # Add a dense layer of 128 nodes, then ReLU Activation:
        nn.Linear(784, 128),
        nn.ReLU(),

        # Add a dense layer of 64 nodes, then ReLU Activation:
        nn.Linear(128, 64),
        nn.ReLU(),

        # Add a dense layer of 10 nodes:
        nn.Linear(64, 10),
    )

    return model

def train_model(model, train_loader, criterion, T):

    # Set the model to training mode:
    model.train()

    # Optimization function:
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    index = 0
    accurate = 0

    # Iterate through the dataset:
    for epoch in range(0, T):
        totalLoss = 0.0
        trainset = get_data_loader(True)

        for i, data in enumerate(trainset, 0):
            index += 1
            inputs, labels = data
            
            opt.zero_grad()
            outputs = model(inputs)
            
            # Backwards + step:
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step

            totalLoss += loss.item()

            _, predict = torch.max(outputs.data, 1)
            #totalLoss += labels.size(0)
            accurate += (predict == labels).sum().item()
            
        # Print the accuracy and loss stats:
        length =  len(trainset.dataset)
        printThings = [epoch, accurate, length, round((accurate/length)*100, 2), round(totalLoss/index, 3)]
        print('\n Train Epoch: {} Accuracy: {}/{}({}%) Loss: {}'.format(*printThings))
        totalLoss = 0.0

def evaluate_model(model, test_loader, criterion, show_loss = True):

    index = 0
    correct = 0
    total = 0

    model.eval()

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(correct/total)

def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
