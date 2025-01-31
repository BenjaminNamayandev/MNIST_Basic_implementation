import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# Create training dataset
train_data = datasets.MNIST(
    root = 'data',
    train = True,
    transform = ToTensor(),
    download=True
)

# Creat Testing DataSet
test_data = datasets.MNIST(
    root = 'data',
    train = False,
    transform = ToTensor(),
    download=True
)

loaders = {
    'train': DataLoader(train_data,
                        batch_size=100,
                        shuffle=True,
                        num_workers = 1),
    
    'test': DataLoader(test_data,
                        batch_size=100,
                        shuffle=True,
                        num_workers = 1)
}

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # convs
        self.conv1 = nn.Conv2d(1, 10, kernel_size = 5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size = 5)
        self.conv2_drop = nn.Dropout2d()

        #fcs
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # pass data into the first layer, pool it, get activation, rinse and repeat
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) # Different types of pooling extracts different data: max = most prominent features, mean = more for spacial, max = takes average of entire map, good for classifications and large, abstract patterns
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)) ## Same thing with a drop-out layer
        
        # Resphape data
        x = x.view(-1, 320) # 20 channels, 4x4 is the number of times the kernels can pass through the dataset, -1 fits tell pytorch to determine the batch size
        x = F.relu(self.fc1(x)) # pass the linear into the relu function
        x = F.dropout(x, training=self.training) # only active during training
        x = self.fc2(x)

        return F.softmax(x) # gets probability of a certain output being a digit
    

  

def train(epoch):
    model.train() # train Mode
    for i, (data, target) in enumerate(loaders['train']):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # clear grad
        output = model(data) # equivalent to model.forward
        loss = loss_fn(output, target) # loss fn
        loss.backward() # backprop w loss function
        optimizer.step() # take a step by lr towards minima
        
        if i % 20 == 0:
            print(f'Train Epoch: {epoch} [{i * len(data)}/{len(loaders["train"].dataset)} ({100. *  i/len(loaders["train"]):.0f}%)]\t{loss.item():.6f}')


def test():
    model.eval() # test mode

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in loaders['test']:
            data, target = data.to(device), target.to(device)
            output = model(data) # equivalent to model.forward(data)

        ####Logging Stuff
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loaders['test'].dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy {correct}/{len(loaders["test"].dataset)} ({100. *correct/len(loaders["test"].dataset):.0f}%\n)')
    ###End of logging


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
model = Net().to(device) # Creates model object
optimizer = optim.Adam(model.parameters(), lr=0.001) # sets an optimizer function with the parameters, and the "step-size". the model.parameters() sector retrieve all trainable parameters (layers) in the model
loss_fn = nn.CrossEntropyLoss() # defined the loss function

if __name__ == '__main__':
    for epoch in range(1,11): #train model 10 times, and then test
        train(epoch)
        test()