import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn import metrics


try:
    train_data = pd.read_csv("/Users/sharathkarnati/Desktop/ML/KARNATI-011852253/Assignment 2/data/fashion-mnist_train.csv")
    test_data = pd.read_csv("/Users/sharathkarnati/Desktop/ML/KARNATI-011852253/Assignment 2/data/fashion-mnist_test.csv")
    print("Data loaded successfully!")
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()


print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")


train_X = train_data.iloc[:, 1:].values
train_Y = train_data.iloc[:, 0].values
test_X = test_data.iloc[:, 1:].values
test_Y = test_data.iloc[:, 0].values

X_train, X_validate, y_train, y_validate = train_test_split(train_X, train_Y, test_size=0.20, random_state=42, shuffle=True)
print(f"Training set shape: {X_train.shape}, Validation set shape: {X_validate.shape}")


print("Scaling training and validation data...")
X_train = preprocessing.scale(X_train)
X_validate = preprocessing.scale(X_validate)
test_X = preprocessing.scale(test_X)  # Ensure the test data is scaled too
print("Data scaling complete.")

C_values = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
train_accuracy = []
val_accuracy = []
test_accuracy = []
support_vectors = []

for C in C_values:
    print(f"Training with C={C}...")
    svm_classifier = SVC(kernel='linear', C=C, max_iter=3000)
    svm_classifier.fit(X_train, y_train)
    print(f"Model trained with C={C}")

   
    svm_train_predictions = svm_classifier.predict(X_train)
    train_acc = metrics.accuracy_score(y_train, svm_train_predictions) * 100
    train_accuracy.append(train_acc)
    print(f"Training accuracy with C={C}: {train_acc:.2f}%")

   
    svm_validate_predictions = svm_classifier.predict(X_validate)
    val_acc = metrics.accuracy_score(y_validate, svm_validate_predictions) * 100
    val_accuracy.append(val_acc)
    print(f"Validation accuracy with C={C}: {val_acc:.2f}%")

  
    svm_test_predictions = svm_classifier.predict(test_X)
    test_acc = metrics.accuracy_score(test_Y, svm_test_predictions) * 100
    test_accuracy.append(test_acc)
    print(f"Test accuracy with C={C}: {test_acc:.2f}%")

    
    support_vectors.append(np.sum(svm_classifier.n_support_))
    print(f"Support vectors with C={C}: {np.sum(svm_classifier.n_support_)}")


print("Final Results:")
print("Training accuracy: ", train_accuracy)
print("Validation accuracy: ", val_accuracy)
print("Test accuracy: ", test_accuracy)

plt.figure(figsize=(10, 6))
plt.plot(C_values, train_accuracy, label='Training Accuracy')
plt.plot(C_values, val_accuracy, label='Validation Accuracy')
plt.plot(C_values, test_accuracy, label='Test Accuracy')
plt.xlabel('C')
plt.ylabel('Accuracy (in percentage)')
plt.legend()
plt.title('SVM with Linear Kernel - Accuracy vs. C')
plt.xscale('log')
plt.ylim(50, 100)


output_path_accuracy = '/Users/sharathkarnati/Desktop/ML/KARNATI-011852253/Assignment 2/output/1.a_accuracy_graph.png'
plt.savefig(output_path_accuracy)
print(f"Accuracy plot saved as {output_path_accuracy}")
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(C_values, support_vectors, label='Support Vectors')
plt.xlabel('C (log scale)')
plt.ylabel('Number of Support Vectors')
plt.legend()
plt.title('SVM with Linear Kernel - Support Vectors vs. C')
plt.xscale('log')


output_path_vectors = '/Users/sharathkarnati/Desktop/ML/KARNATI-011852253/Assignment 2/output/1.a_support_vectors.png'
plt.savefig(output_path_vectors)
print(f"Support vector plot saved as {output_path_vectors}")
plt.show()
