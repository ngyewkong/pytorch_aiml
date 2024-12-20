# calculate dot product using code
# %% packages
import numpy as np
# %%
# X in input vector, w1 & w2 are weights
X = [0, 1]
w1 = [2, 3]
w2 = [0.4, 1.8]
# %%
# manual calculation of dot product
dot_X_w1 = X[0] * w1[0] + X[1] * w1[1]
dot_X_w2 = X[0] * w2[0] + X[1] * w2[1]

# larger values indicates a stronger similarity to the input vector
print(dot_X_w1)  # 3
print(dot_X_w2)  # 1.8
# %%
# using numpy to calculate dot product
np_dot_X_w1 = np.dot(X, w1)
np_dot_X_w2 = np.dot(X, w2)

print(np_dot_X_w1)  # 3
print(np_dot_X_w2)  # 1.8
# %%
