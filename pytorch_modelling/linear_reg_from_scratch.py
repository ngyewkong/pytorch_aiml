# %% packages
import os
from sklearn.linear_model import LinearRegression
from torchviz import make_dot
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sns

# %% data import
cars_file = 'https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv'
cars = pd.read_csv(cars_file)
cars.head()

# %% visualise the model
sns.scatterplot(x='wt', y='mpg', data=cars)
sns.regplot(x='wt', y='mpg', data=cars)

# %% convert data to tensor
cars_x_list = cars.wt.values
cars_x_np = np.array(cars_x_list, dtype=np.float32).reshape(-1, 1)
print(cars_x_np)
print(cars_x_np.shape)

cars_y_list = cars.mpg.values.tolist()

# data in numpy data type -> convert to tensor
X = torch.from_numpy(cars_x_np)

# data in list data type
y = torch.tensor(cars_y_list)
# %% training
# create weights (weights are randomly initialized)
w = torch.rand(1, requires_grad=True, dtype=torch.float32)  # 1 value

# create bias
b = torch.rand(1, requires_grad=True, dtype=torch.float32)

# training parameters
epochs = 1000
learning_rate = 0.001

# iterate across the epochs
for epoch in range(epochs):
    for i in range(len(X)):  # passing batch size of 1 observation
        # calculate the y prediction (forward pass)
        # which is just the X observation multiply by weight plus bias
        y_pred = X[i] * w + b

        # calculate loss
        # (y_pred - y_actual)^2
        loss_tensor = torch.pow(y_pred - y[i], 2)

        # backward pass
        loss_tensor.backward()

        # extract losses
        loss_value = loss_tensor.data[0]

        # update the weights and biases
        with torch.no_grad():
            # in this block we are deactivating the autogradient
            # make use of the calculated gradient
            # and zero-ing the gradient so the next epoch can start
            w -= w.grad * learning_rate
            b -= b.grad * learning_rate

            w.grad.zero_()  # zero_ inplace operation no need to do assignment
            b.grad.zero_()
    # check the loss value on each epoch run
    print(f"Epoch {epoch} loss: {loss_value}")  # should see a decreasing loss


# %% check results
print(f"Weight: {w.item()}")
print(f"Bias: {b.item()}")
# %%
y_pred = ((X * w) + b).detach().numpy()
# Converting to NumPy: PyTorch requires you to explicitly strip away the computational history before you can cast a GPU/tracked tensor into a standard CPU NumPy array.

sns.scatterplot(x=cars_x_list, y=cars_y_list)
# rmb to remove 1 dimension from y_pred
sns.lineplot(x=cars_x_list, y=y_pred.reshape(-1))
# %% (Statistical) Linear Regression
reg = LinearRegression().fit(cars_x_np, cars_y_list)
print(
    f"SKLearn Linear Regression Slope: {reg.coef_}, Intercept: {reg.intercept_}")
print(f"Linear Model from Scratch Slope: {w.item()}, Intercept: {b.item()}")
# should see very close to the linear model results from sklearn

# %% create graph visualisation
# make sure GraphViz is installed (https://graphviz.org/download/)
# if not computer restarted, append directly to PATH variable
make_dot(loss_tensor)
# %%
