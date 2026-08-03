# %% packages

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
X_list = cars.wt.values
X_np = np.array(X_list, dtype=np.float32).reshape(-1, 1)
y_list = cars.mpg.values
y_np = np.array(y_list, dtype=np.float32).reshape(-1, 1)
X = torch.from_numpy(X_np)
y_true = torch.from_numpy(y_np)

# %% model class
# class is inheriting from nn.Module


class LinearRegressionTorch(nn.Module):
    # need to have __init__ function
    def __init__(self, input_size, output_size):

        # super() creates a proxy object that delegates method calls to the next class in the method resolution order (MRO)
        # finds superclass of LinearRegressionTorch in the MRO
        # binds that superclass method to self
        # then calls on nn.Module.__init__(self)
        # super() is needed when a class inherits from another class and you want the parent class to do its own initialization or handle a method call too.
        # In plain English
        # super() is a way to say: “run the parent class version of this method”
        # MRO is the rulebook Python uses to decide which parent class to call when there are multiple possibilities
        # In your code, super().__init__() means: “call the __init__ defined in nn.Module so the base PyTorch module is properly initialized.”
        super(LinearRegressionTorch, self).__init__()
        # creating an internal object called linear
        self.linear = nn.Linear(input_size, output_size)

    # forward pass
    def forward(self, x):
        out = self.linear(x)
        return out


# set the input & output dimensions
input_dim = 1
output_dim = 1

# init the model
model = LinearRegressionTorch(input_size=input_dim, output_size=output_dim)
print(model)

print(hasattr(model, "named_parameters"))
print(model.named_parameters)

# print the named parameters
print("----- Model's Named Parameters -----")
for name, param in model.named_parameters():
    print(name, param.shape, param.detach().numpy())

# %% loss function

# using Mean Squared Error Loss from torch.nn
loss_func = nn.MSELoss()

# %% optimizer
LR = 0.02  # test different values 0.001 - 0.1 best is 0.02
optimizer = torch.optim.SGD(model.parameters(), lr=LR)


# %% perform training

losses, slope, bias = [], [], []

# test with different epoch value (1000, 10000, 100000)
NUM_EPOCHS = 10000

for epoch in range(NUM_EPOCHS):
    # set gradients to zeros
    optimizer.zero_grad()

    # forward pass
    y_pred = model(X)

    # compute loss
    loss = loss_func(y_pred, y_true)

    # backprop
    loss.backward()

    # update weights
    optimizer.step()

    # get parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name == 'linear.weight':
                slope.append(param.data.numpy()[0][0])  # the very first object
            if name == 'linear.bias':
                bias.append(param.data.numpy()[0])

    # store loss
    losses.append(float(loss.data))

    # print loss
    if epoch % 100 == 0:
        print("Epoch: {}, Loss: {:.4f}".format(epoch, loss.data))


# %% visualise model training
sns.scatterplot(x=range(NUM_EPOCHS), y=losses)

# %% visualise the bias development
sns.scatterplot(x=range(NUM_EPOCHS), y=bias)

# %% visualise the slope development
sns.scatterplot(x=range(NUM_EPOCHS), y=slope)

# %% check the result
y_pred = model(X).data.numpy().reshape(-1)

sns.scatterplot(x=X_list, y=y_list)  # actual data plot
sns.scatterplot(x=X_list, y=y_pred, color='red')  # the linear regression line

# %%
# bias: 37.3, slope: -5.3
print(f"Bias: {bias}\n")
print(f"Slope: {slope}\n")
# %%
