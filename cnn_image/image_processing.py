# %%
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
# %% import image
img = Image.open('kiki.jpg')
img

# %% check size of image
img.size
# %% compose a series of steps
preprocess_steps = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomRotation(50),
    transforms.CenterCrop(200),
    transforms.Grayscale(),
    transforms.RandomVerticalFlip(),
    # after happy with the different transformations applied then convert back to Tensor
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),  # improve processing load
])

x = preprocess_steps(img)

# %% check size after preprocessing
# from 768, 1024 -> 300, 400 (scaled based on the first size reduction) or set a tuple (300, 300)
x.shape
# %%
x
# %% how to get mean & std from the given image
x.mean(), x.std()

# %%
