# https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lib.keycol import keycol

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    # x represents our data
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)

        # Run max pooling over x
        x = F.max_pool2d(x, 2)

        # Pass data through dropout1
        x = self.dropout1(x)

        # Flatten x with start_dim=1
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = F.relu(x)

        x = self.dropout2(x)
        x = self.fc2(x)
        
        output = F.log_softmax(x, dim=1)
        return output

def main():
    model = Net()
    torch.set_printoptions(sci_mode=False)

    #STDOUT

    print()

    print(model)
    print()

    x = torch.rand((1, 1, 28, 28))
    print(x)
    print()

    y = model(x)
    print(y)
    print()

    print(y.shape)

    # print()
    # arr = filter(lambda x: x > 3, torch.randn(1000))
    # print(list(arr))

main()