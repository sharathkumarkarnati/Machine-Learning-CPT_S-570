import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def polynomial_kernel(x_train, x_j, p):
    return (1 + np.matmul(x_train, np.reshape(x_j, (x_train.shape[1], 1)))) ** p

def predict_y(alpha_m, x, x_train, k):
    predicted_vector = np.matmul(alpha_m, polynomial_kernel(x_train, x, p))
    return np.argmax(predicted_vector)

def kernelized_perceptron_train(x_train, y_train, k, T):
    alpha_m = np.zeros((k, len(x_train)))
    mistakes = []

    for i in range(T):
        count = 0
        for j in range(len(x_train)):
            y_hat = predict_y(alpha_m, x_train[j], x_train, k)
            if y_hat != y_train[j]:
                alpha_m[int(y_hat), j] -= 1
                alpha_m[int(y_train[j]), j] += 1
                count += 1

        mistakes.append(count)
        print(f'Iteration {i+1}: Mistakes = {count}')

    return alpha_m, mistakes


train_data = pd.read_csv("/Users/sharathkarnati/Desktop/ML/KARNATI-011852253/Assignment 2/data/fashion-mnist_train.csv")
test_data = pd.read_csv("/Users/sharathkarnati/Desktop/ML/KARNATI-011852253/Assignment 2/data/fashion-mnist_test.csv")


train_X = train_data.iloc[:, 1:].values
train_Y = train_data.iloc[:, 0].values
test_X = test_data.iloc[:, 1:].values
test_Y = test_data.iloc[:, 0].values


x_train = train_X[0:1000]
y_train = train_Y[0:1000]
x_validation = train_X[1000:1200]
y_validation = train_Y[1000:1200]


k = 10  
T = 5   
p = 2   


alpha_m, mistakes = kernelized_perceptron_train(x_train, y_train, k, T)

# Calculate training, validation, and testing accuracy
train_predictions = [predict_y(alpha_m, x, x_train, k) for x in x_train]
val_predictions = [predict_y(alpha_m, x, x_train, k) for x in x_validation]
test_predictions = [predict_y(alpha_m, x, x_train, k) for x in test_X]

train_accuracy = accuracy_score(y_train, train_predictions) * 100
val_accuracy = accuracy_score(y_validation, val_predictions) * 100
test_accuracy = accuracy_score(test_Y, test_predictions) * 100

print(f"Training Accuracy: {train_accuracy:.2f}%")
print(f"Validation Accuracy: {val_accuracy:.2f}%")
print(f"Testing Accuracy: {test_accuracy:.2f}%")


plt.plot(range(1, T + 1), mistakes, marker='o')
plt.xlabel('Training Iterations')
plt.ylabel('Number of Mistakes')
plt.title('Mistakes during Kernelized Perceptron Training')
plt.xticks(range(1, T + 1))
plt.grid()
plt.savefig('/Users/sharathkarnati/Desktop/ML/KARNATI-011852253/Assignment 2/output/2.a_mistakes.jpeg')
plt.show()
