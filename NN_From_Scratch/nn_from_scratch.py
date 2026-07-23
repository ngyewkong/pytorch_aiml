# %% packages
from collections import Counter
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# %% data prep
# source: https://www.kaggle.com/datasets/iamsouravbanerjee/heart-attack-prediction-dataset
df = pd.read_csv("heart.csv")
df.head()

# %% separate independent & dependent variables/features
X = np.array(df.loc[:, df.columns != 'output'])
y = np.array(df['output'])

print(f"X: {X.shape}, y: {y.shape}")
# %% Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123)

# %% scale the data
# normal distribution scaling as the range variance is quite big
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# transform no need for fit_transform -> use directly the scaling factor from X_train
X_test_scaled = scaler.transform(X_test)

# %% neural network class


class NeuralNetworkFromScratch:
    def __init__(self, LR, X_train, y_train, X_test, y_test):
        # initialize the weights & biases
        # initial weights are randomised with the number of features of X_train
        self.w = np.random.randn(X_train.shape[1])
        self.b = np.random.randn()  # only one bias (y = w*X + b)
        self.LR = LR
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.L_train = []  # training set loss
        self.L_test = []  # testing set loss

    # helper functions
    def activation(self, x):
        # sigmoid function
        return 1 / (1 + np.exp(-x))

    def derivative_activation(self, x):
        # the derivative of the sigmoid function
        return self.activation(x) * (1 - self.activation(x))

    # forward pass
    def forward(self, X):
        # the dot product of the input and the weight + bias
        hidden_1 = np.dot(X, self.w) + self.b
        activate_1 = self.activation(hidden_1)
        return activate_1

    # backward pass
    # take the input and y_true values to calculate the gradient
    def backward(self, X, y_true):
        # calc gradients
        hidden_1 = np.dot(X, self.w) + self.b
        y_pred = self.forward(X)
        derivativeLoss_predictions = 2 * (y_pred - y_true)
        derivativePrediction_hidden_1 = self.derivative_activation(hidden_1)
        derivativeHidden1_bias = 1
        derivativeHidden1_weight = X

        derivativeLoss_bias = derivativeLoss_predictions * \
            derivativePrediction_hidden_1 * derivativeHidden1_bias
        derivativeLoss_weight = derivativeLoss_predictions * \
            derivativePrediction_hidden_1 * derivativeHidden1_weight

        return derivativeLoss_bias, derivativeLoss_weight

    # optimizer function
    def optimizer(self, derivativeLoss_bias, derivativeLoss_weight):
        # update weights & biases with the new derivativeLoss of each component * learning rate
        self.b = self.b - (derivativeLoss_bias * self.LR)
        self.w = self.w - (derivativeLoss_weight * self.LR)

    # train function
    def train(self, iterations):
        for i in range(iterations):
            # random position
            random_pos = np.random.randint(len(self.X_train))

            # forward pass
            y_train_true = self.y_train[random_pos]
            y_train_pred = self.forward(self.X_train[random_pos])

            # calculate training losses
            training_loss = np.sum(np.square(y_train_pred - y_train_true))
            self.L_train.append(training_loss)

            # calculate gradients
            derivativeLoss_bias, derivativeLoss_weight = self.backward(
                self.X_train[random_pos], y_train[random_pos])

            # update weights
            self.optimizer(derivativeLoss_bias, derivativeLoss_weight)

            # calculate error for test data
            Loss_sum = 0
            for j in range(len(self.X_test)):
                y_true = self.y_test[j]
                y_pred = self.forward(self.X_test[j])
                Loss_sum += np.square(y_pred - y_true)
            self.L_test.append(Loss_sum)

        return "Training successful"


# %% setup hyperparameters (LR, ITERATIONS)
LR = 0.1
ITERATIONS = 1000

# %% initialize model instance and training
nn = NeuralNetworkFromScratch(
    LR=LR, X_train=X_train_scaled, y_train=y_train, X_test=X_test_scaled, y_test=y_test)

nn.train(iterations=ITERATIONS)

# %% model evaluation
# check losses
sns.lineplot(x=list(range(len(nn.L_test))), y=nn.L_test)

# %% iterate over test data
total = X_test_scaled.shape[0]
correct = 0
y_preds = []

for i in range(total):
    y_true = y_test[i]
    # round the value of y_pred
    y_pred = np.round(nn.forward(X_test_scaled[i]))  # from forward pass
    y_preds.append(y_pred)
    correct += 1 if y_true == y_pred else 0

# %% Calculate Accuracy
accuracy = (correct / total) * 100
print("The model accuracy is ", accuracy, "%")
# %% Baseline Classifier
Counter(y_test)  # used to check the class labels percentage in the testing dataset
# 31 with label 1, 30 with label 0 --> no class imbalance
# baseline classifier --> predicting the class with the highest distribution
# baseline classifier percentage (just by predicting the majority class)
print(31 / len(y_test))

# trained classifer is outperforming baseline (77% vs 51%)
# %% Confusion Matrix
confusion_matrix(y_true=y_test, y_pred=y_preds)

# 23 7
# 7 24
# 23+24 = 47 correct predictions out of 61 observations

# %%
