# %% packages
from typing import OrderedDict
import torch
import torch.nn as nn

# %% sample input data of certain shape

# tensor dimensions
# dim 1 - batch size -> labels/predictions eg. [16]
# dim 2 - batch size, features -> nn.Linear() eg. [16, 512]
# dim 3 - batch size, channels, features -> nn.Conv1d() eg. [16, 1 512]
# dim 4 - batch size, channels, height, width -> nn.Conv2d() eg. [16, 1, 224, 224]
# dim 5 - batch size, channels, depth, height, width -> nn.Conv3d() eg. [16, 1, 5, 224, 224]
# %%
# in Conv2d()
# in_channels -> number of rgb channels (1 for grayscale, 3 for rgb)
# conv layers get more channels at beginning of network
# out_channels of prev conv layer = in_channel for next conv layer
# Conv2d -> 4d shape (16, 48, 224, 224) - output to fully connected layer which is 2d shape (need to change the tensor size)
# use nn.Flatten() -> 16, 48 * 224 * 224)
# fully connected layer take features not channels
# in_features -> input sample size
# out_features -> output sample size

# %% sample input data of certain shape
input = torch.rand((1, 3, 32, 32))

# input is (1, 3, 32, 32)
model = nn.Sequential(OrderedDict([
    ('conv1', nn.Conv2d(3, 8, 3)),  # out: (BS, 8, 30, 30)
    ('relu1', nn.ReLU()),
    ('pool', nn.MaxPool2d(2, 2)),  # out: (BS, 8, 15, 15)
    ('conv2', nn.Conv2d(8, 16, 3)),  # out: (BS, 16, 13, 13)
    ('relu2', nn.ReLU()),
    ('pool2', nn.MaxPool2d(2, 2)),  # out: (BS, 16, 6, 6)
    ('flatten', nn.Flatten()),  # out: (BS, 16*6*6 = 576)
    ('fc1', nn.Linear(16 * 6 * 6, 128)),  # input: 576, output: 128
    ('relu3', nn.ReLU()),
    ('fc2', nn.Linear(128, 64)),  # input: 128, output: 64
    ('relu4', nn.ReLU()),
    ('fc3', nn.Linear(64, 1)),  # input: 64, output: 1
    ('sigmoid', nn.Sigmoid())  # out: (BS, 1 neuron)
]))

# %% test the model setup
print(model(input).shape)

# %%
