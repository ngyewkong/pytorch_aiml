# %% packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import seaborn as sns

# %% data import
cars_file = 'https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv'
cars = pd.read_csv(cars_file)
cars.head()

# %% visualise the model
sns.scatterplot(x='wt', y='mpg', data=cars)
sns.regplot(x='wt', y='mpg', data=cars)

# %% convert data to tensor
X_list = cars.wt.values
X_np = np.array(X_list, dtype=np.float32).reshape(-1, 1)
y_list = cars.mpg.values
y_np = np.array(y_list, dtype=np.float32).reshape(-1, 1)
X = torch.from_numpy(X_np)
y_true = torch.from_numpy(y_np)

# %% create dataset & dataloader


class LinearRegressionDataset(Dataset):
    # initialize the internal objects X & y
    def __init__(self, X, y):
        self.X = X
        self.y = y

    # return the len of the dataset
    def __len__(self):
        return len(self.X)

    # return the independent features when pass in some index
    def __getitem__(self, index):
        return self.X[index], self.y[index]


# DataLoader class take in the dataset (which is using the LinearRegressionDataset class and taking the numpy X & y var as inputs)
# batch size 2
train_loader = DataLoader(
    dataset=LinearRegressionDataset(X_np, y_np), batch_size=2)


# %% model
class LinearRegressionTorch(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionTorch, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


input_dim = 1
output_dim = 1
model = LinearRegressionTorch(input_size=input_dim, output_size=output_dim)
model.train()

# %% Mean Squared Error
loss_func = nn.MSELoss()

# %% Optimizer
learning_rate = 0.02
# test different values of too large 0.1 and too small 0.001
# best 0.02
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# %% check the train loader
for i, (X, y) in enumerate(train_loader):
    print(f"number {i} batch")
    # [2, 1] which is the batch size of 2, and number of feature 1
    print(f"X size: {X.shape}")
    print(f"X: {X}")
    print(f"y size: {y.shape}")
    print(f"y: {y}")

# %% perform training
losses = []
slope, bias = [], []
NUM_EPOCHS = 1000
BATCH_SIZE = 2

# update to use DataLoader
# using DataLoader we do not need to manually track the i & Batch size
# for our each epoch pass
for epoch in range(NUM_EPOCHS):
    for i, (X, y_true) in enumerate(train_loader):
        # optimization
        optimizer.zero_grad()

        # forward pass
        y_pred = model(X)

        # compute loss
        loss = loss_func(y_pred, y_true)
        losses.append(loss.item())

        # backprop
        loss.backward()

        # update weights
        optimizer.step()

    # get parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name == 'linear.weight':
                slope.append(param.data.numpy()[0][0])
            if name == 'linear.bias':
                bias.append(param.data.numpy()[0])

    # store loss
    losses.append(float(loss.data))
    # print loss
    if (epoch % 100 == 0):
        print(f"Epoch {epoch}, Loss: {loss.data}")


# %% visualise model training
sns.scatterplot(x=range(len(losses)), y=losses)

# %% visualise the bias development
sns.lineplot(x=range(NUM_EPOCHS), y=bias)
# %% visualise the slope development
sns.lineplot(x=range(NUM_EPOCHS), y=slope)


# %% check the result
model.eval()
y_pred = [i[0] for i in model(X).data.numpy()]
y = [i[0] for i in y_true.data.numpy()]
sns.scatterplot(x=X_list, y=y)
sns.lineplot(x=X_list, y=y_pred, color='red')
# %%
