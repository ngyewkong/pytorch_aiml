# %% packages
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
os.getcwd()
print(os.getcwd())

# %% transform, load data
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

batch_size = 4

# set train test set
trainset = torchvision.datasets.ImageFolder(
    root="binary_classification/data/train", transform=transform)  # relative path

testset = torchvision.datasets.ImageFolder(
    root="binary_classification/data/test", transform=transform)

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)

# %% visualize images


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images, nrow=2))
# %% Neural Network setup


class ImageClassificationNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # input image is 1 channel, 32 x 32
        # 1 input channel as image is grayscale
        # 6 output channels
        # kernel size of 3
        self.conv1 = nn.Conv2d(1, 6, 3)  # output: 6 channels, 30 x 30
        self.pool = nn.MaxPool2d(2, 2)  # output: 6 channels, 15 x 15
        # 6 input channels
        # 16 output channels
        # kernel size of 3
        self.conv2 = nn.Conv2d(6, 16, 3)  # output: 16 channels, 13 x 13
        # after next pool: 16, 6 x 6 which is the input size for the linear layer
        self.fc1 = nn.Linear(16 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        # activation is sigmoid since is binary classification
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)  # put input into first conv layer
        x = F.relu(x)  # add rectified linear unit
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)  # add rectified linear unit
        x = self.pool(x)
        # reducing the 4 dimensional tensor to a 2d tensor
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)  # add rectified linear unit
        x = self.fc2(x)
        x = F.relu(x)  # add rectified linear unit
        x = self.fc3(x)
        x = self.sigmoid(x)  # final activation function before output
        return x


# %% init model
model = ImageClassificationNet()
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.8)
# %% training
NUM_EPOCHS = 10
for epoch in range(NUM_EPOCHS):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        # zero gradients
        optimizer.zero_grad()
        # forward pass
        outputs = model(inputs)
        # calc losses
        # ensure outputs & labels are the same dimensions
        loss = loss_fn(outputs, labels.reshape(-1, 1).float())
        # backward pass
        loss.backward()
        # update weights
        optimizer.step()

        if i % 100 == 0:
            print(f'Epoch {epoch}/{NUM_EPOCHS}, Step {i+1}/{len(train_loader)},'
                  f'Loss: {loss.item():.4f}')
# %% test
y_test = []
y_test_pred = []
for i, data in enumerate(test_loader, 0):
    inputs, y_test_temp = data
    with torch.no_grad():
        y_test_hat_temp = model(inputs).round()

    y_test.extend(y_test_temp.numpy())
    y_test_pred.extend(y_test_hat_temp.numpy())

# %%
acc = accuracy_score(y_test, y_test_pred)
print(f'Accuracy: {acc*100:.2f} %')  # 86.67%
# %%
# We know that data is balanced, so baseline classifier has accuracy of 50 %.
