import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as layers
from torch import optim
from torch import tensor

# Set environment variable to avoid any errors with library usage
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Prepare the data:
# Converting the train data and test data to map with pytorch tensor
def preparedata():
    train_set = pd.read_csv('/Users/sharathkarnati/Desktop/ML/KARNATI-011852253/Assignment 4/data/fashion-mnist_train.csv')
    test_set = pd.read_csv('/Users/sharathkarnati/Desktop/ML/KARNATI-011852253/Assignment 4/data/fashion-mnist_test.csv')
    train_data = train_set[train_set.columns[1:]].values
    train_label = train_set.label.values
    test_data = test_set[test_set.columns[1:]].values
    test_label = test_set.label.values
    return map(tensor, (train_data, train_label, test_data, test_label))

# Inputting the data to the variables
train_data, train_label, test_data, test_label = preparedata()

# Getting the shape feature available for the test and train data
train_n, train_m = train_data.shape
test_n, test_m = test_data.shape
n_cls = train_label.max() + 1

# Plot to understand the type of data we are dealing with
plt.imshow(train_data[torch.randint(train_n, (1,))].view(28, 28))  # visualize a random image in the training data
plt.show()

# Define the CNN model:
class CNNModel(layers.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Defining the layers according to the Question requirements
        self.conv1 = layers.Conv2d(1, 8, 5, 2, 2)
        self.conv2 = layers.Conv2d(8, 16, 3, 2, 1)
        self.conv3 = layers.Conv2d(16, 32, 3, 2, 1)
        self.conv4 = layers.Conv2d(32, 32, 3, 2, 1)
        self.fc1 = layers.Linear(32 * 1 * 1, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        m = layers.AdaptiveAvgPool2d(1)
        x = m(F.relu(self.conv4(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# Create the model:
torch.manual_seed(1000)
model = CNNModel()
lr = 0.005
epochs = 10
bs = 32
loss_func = F.cross_entropy
opt = optim.Adam(model.parameters(), lr=lr)
train_accuracy_vals = []
test_accuracy_vals = []

# Train & Test the model for 10 Epochs:
for iter in range(epochs):
    model.train()
    for i in range((train_n - 1) // bs + 1):
        start_i = i * bs
        end_i = start_i + bs
        xb = train_data[start_i:end_i].float().reshape(bs, 1, 28, 28)
        yb = train_label[start_i:end_i]
        loss = loss_func(model(xb), yb)
        loss.backward()
        opt.step()
        opt.zero_grad()

    model.eval()
    with torch.no_grad():
        total_loss, train_accuracy, test_accuracy = 0., 0., 0.
        for i in range(train_n):
            x = train_data[i].float().reshape(1, 1, 28, 28)
            y = train_label[i]
            pred = model(x)
            train_accuracy += (torch.argmax(pred) == y).float()
        train_accuracy_vals.append((train_accuracy * 100.0 / train_n).item())
        
        for i in range(test_n):
            x = test_data[i].float().reshape(1, 1, 28, 28)
            y = test_label[i]
            pred = model(x)
            test_accuracy += (torch.argmax(pred) == y).float()
        test_accuracy_vals.append((test_accuracy * 100.0 / test_n).item())
        
        print(f"Iteration: {iter + 1}")
        print(f"Fashion-Mnist Training Accuracy: {train_accuracy * 100.0 / train_n:.2f}%")
        print(f"Fashion-Mnist Testing Accuracy: {test_accuracy * 100.0 / test_n:.2f}%")

# Plotting the accuracy for both training and testing
epochs_range = list(range(1, epochs + 1))  # List of epoch numbers [1, 2, 3, ..., 10]

plt.plot(epochs_range, train_accuracy_vals, label='Training Accuracy')
plt.plot(epochs_range, test_accuracy_vals, label='Testing Accuracy')

plt.title("Training and Testing Accuracy over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")

# Set x-axis to show values 1 through 10
plt.xticks(epochs_range)

# Set y-axis ticks for better readability
plt.yticks([82, 84, 86, 88, 90])

plt.legend()
plt.grid(True)
plt.show()
