# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

from random import randint

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

# FETCH DATASET

train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
batch_size = 64

# LOAD DATASET

# Returns array of ceil(60,000 / 64) == 938 tuples, each containing a tensor of 64 images and a tensor of 64 labels.
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True) 
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

""" for x, y in test_dataloader:
    print(f"Shape of x [N, C, H, W]: {x.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break """

labels_map = [
    "T-Shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot",
]

# DISPLAY DATASET

def display_data():
    figure = plt.figure(figsize=(8, 2))
    for x, y in train_dataloader:
        for i in range(10):
            img = x[i].squeeze()
            label = y[i].item()

            plt.subplot(1, 10, i+1)
            plt.imshow(img, cmap="gray")
            plt.title(labels_map[label])
            plt.axis("off")
        plt.show()
        break

# CONFIGURE DEVICE (CPU / GPU)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# print(f"Using {device} device")

# BUILD MODEL

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# TRAIN, TEST, AND SAVE MODEL

def init_model():
    """
        # TERMS
        
        # epoch = 1 full forward pass = 60,000 samples (img/label pairs)
        # batch_size = 64 samples
        # batches (SGD iterations) = 60,000 / 64 = 938
    """
    
    model = Model().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    def train(dataloader, model, loss_fn, optimizer):
        model.train()
        size = len(dataloader.dataset)

        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            # Loss (prediction error)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(x)
                print(f"loss: {loss:>7f}% [{current:>5d}/{size:>5d}]")
    
    def test(dataloader, model, loss_fn):
        model.eval()
        size = len(dataloader.dataset)
        batches = len(dataloader)

        test_loss, correct = 0, 0

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                test_loss += loss_fn(y_pred, y).item()
                correct += (y_pred.argmax(dim=1) == y).type(torch.float).sum().item()

            test_loss /= batches
            correct /= size
            print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}% \n")
    
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-----------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

# LOAD MODEL

def load_model(model_path, test_sample):
    model = Model().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    x, y = test_data[test_sample][0], test_data[test_sample][1]
    with torch.no_grad():
        x = x.to(device)
        y_pred = model(x)
        predicted, actual = labels_map[y_pred[0].argmax(dim=0)], labels_map[y]
        print(f"\nSample #{test_sample}")
        print(f"Predicted: {predicted} \nActual: {actual}")

# display_data()
init_model()
# load_model("model.pth", randint(0, 100))