# %% package
import torch
import numpy as np
import seaborn as sns

# %% create a tensor
x = torch.tensor(5.5)

# %% simple calculations with tensors
y = x + 10
print(y)  # tensor 15.5 (behaving like how numpy does calculations)
print(y.type)  # Tensor object type
print(x.requires_grad)  # False as by default is set to False

# %%
# only for floating & complex data types can require gradients
z = torch.tensor(12.0, requires_grad=True)
print(z.requires_grad)  # True
# %%


def y_function(val):
    """
    Function to give a value of y for every x input
    """
    return (val - 3) * (val - 6) * (val - 4)


x_range = np.linspace(0, 10, 101)

y_range = [y_function(i) for i in x_range]

sns.lineplot(x=x_range, y=y_range)
# %% calculate gradient

y = (z - 3) * (z - 6) * (z - 4)
y.backward()  # the function to calculate the gradients

print(z.grad)
# %% second eg
z2 = torch.tensor(1.0, requires_grad=True)
y = z2**3
z3 = 5 * z2 - 4
z3.backward()

# %%
print(z2.grad)
# %% complex eg
x11 = torch.tensor(2.0, requires_grad=True)
x21 = torch.tensor(3.0, requires_grad=True)

x12 = 5 * x11 - 3 * x21
x22 = 2 * x11 ** 2 + 2 * x21

# output layer
y = 4 * x12 + 3 * x22
y.backward()
print(x11.grad)
print(x21.grad)

# The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor.
print(x12.grad)
print(x22.grad)

# %%
