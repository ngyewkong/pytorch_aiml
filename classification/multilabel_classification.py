# %% packages
from ast import Mult
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import numpy as np
from collections import Counter
# %% data prep
# 10000 samples, 10 features, 3 classes, 2 labels (a sample can have 0, 1, 2, 3 observations of the classes)
X, y = make_multilabel_classification(
    n_samples=10000, n_features=10, n_classes=3, n_labels=2)
X_torch = torch.FloatTensor(X)
y_torch = torch.FloatTensor(y)

# %% train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_torch, y_torch, test_size=0.2)


# %% dataset and dataloader
class MultilabelDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# TODO: create instance of dataset
train_dataset = MultilabelDataset(X_train, y_train)
test_dataset = MultilabelDataset(X_test, y_test)

# TODO: create train loader & test loader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# %% model
# TODO: set up model class
# topology: fc1, relu, fc2
# final activation function??


class MultiLabelClassifier(nn.Module):
    # init func
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    # forward pass
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


# TODO: define input and output dim
input_dim = train_dataset.X.shape[1]
output_dim = train_dataset.y.shape[1]

print(f"Input Dimensions: {input_dim}")
print(f"Output Dimensions: {output_dim}")

# TODO: create a model instance
model = MultiLabelClassifier(
    input_size=input_dim, hidden_size=20, output_size=output_dim)

# %% loss function, optimizer, training loop
# TODO: set up loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
losses = []
slope, bias = [], []
number_epochs = 100

# TODO: implement training loop
for epoch in range(number_epochs):
    for j, (X, y) in enumerate(train_loader):
        # setting optimizer gradients to zeroes
        optimizer.zero_grad()

        # forward pass
        y_pred = model(X)

        # compute loss
        loss = loss_fn(y_pred, y)

        # backward pass
        loss.backward()

        # update weights
        optimizer.step()

    # TODO: print epoch and loss at end of every 10th epoch
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.data.item()}")
        losses.append(loss.item())


# %% losses
# TODO: plot losses
sns.scatterplot(x=range(len(losses)), y=losses)

# %% test the model
# TODO: predict on test set
# using the model for inference
# set to no_grad to use the model in eval mode
with torch.no_grad():
    y_test_pred = model(X_test).round()


# %% Naive classifier accuracy
# TODO: convert y_test tensor [1, 1, 0] to list of strings '[1. 1. 0.]'
y_test_str = [str(i) for i in y_test.detach().numpy()]
Counter(y_test_str)

# TODO: get most common class count
most_common_count = Counter(y_test_str).most_common()[0][1]

# TODO: print naive classifier accuracy (predicting just the most common observations)
# 21.65%
print(f"Naive Classifier Accuracy: {most_common_count/len(y_test_str) * 100}%")


# %% Test accuracy
# TODO: get test set accuracy
print(
    f"Model Accuracy: {accuracy_score(y_test, y_test_pred) * 100}%"
)  # 73.85%
# %%
