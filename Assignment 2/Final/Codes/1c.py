import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics

# Load the training and test data
train_data = pd.read_csv("/Users/sharathkarnati/Desktop/ML/KARNATI-011852253/Assignment 2/data/fashion-mnist_train.csv")
test_data = pd.read_csv("/Users/sharathkarnati/Desktop/ML/KARNATI-011852253/Assignment 2/data/fashion-mnist_test.csv")

# Prepare the features and labels
train_X = train_data.iloc[:, 1:].values
train_Y = train_data.iloc[:, 0].values
test_X = test_data.iloc[:, 1:].values
test_Y = test_data.iloc[:, 0].values

# Split the training data into training and validation sets
X_train, X_validate, y_train, y_validate = train_test_split(train_X, train_Y, test_size=0.20, random_state=0)

# Find the best C value using a linear kernel
C_values = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
best_C = None
best_val_accuracy = 0

for C in C_values:
    svm_classifier = SVC(kernel='linear', C=C, max_iter=3000)
    svm_classifier.fit(X_train, y_train)
    
    svm_validate_predictions = svm_classifier.predict(X_validate)
    val_accuracy = metrics.accuracy_score(y_validate, svm_validate_predictions) * 100

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_C = C

print(f"Best C value from Linear Kernel: {best_C} with Validation Accuracy: {best_val_accuracy:.2f}%")

# Initialize lists to store accuracies and support vectors for different kernels
results = {
    'Kernel': [],
    'Train Accuracy': [],
    'Validation Accuracy': [],
    'Test Accuracy': [],
    'Support Vectors': []
}

# Function to evaluate a given SVM model
def evaluate_svm(kernel, degree=None):
    svm_classifier = SVC(kernel=kernel, C=best_C, degree=degree, max_iter=3000)
    svm_classifier.fit(X_train, y_train)
    
    train_predictions = svm_classifier.predict(X_train)
    val_predictions = svm_classifier.predict(X_validate)
    test_predictions = svm_classifier.predict(test_X)
    
    results['Kernel'].append(kernel)
    results['Train Accuracy'].append(metrics.accuracy_score(y_train, train_predictions) * 100)
    results['Validation Accuracy'].append(metrics.accuracy_score(y_validate, val_predictions) * 100)
    results['Test Accuracy'].append(metrics.accuracy_score(test_Y, test_predictions) * 100)
    results['Support Vectors'].append(np.sum(svm_classifier.n_support_))

# Evaluate Linear Kernel
evaluate_svm(kernel='linear')

# Evaluate Polynomial Kernels
for degree in [2, 3, 4]:
    evaluate_svm(kernel='poly', degree=degree)

# Convert results to DataFrame for easier visualization
results_df = pd.DataFrame(results)
print(results_df)

# Plotting the results
results_df.set_index('Kernel').plot(kind='bar', figsize=(10, 6), alpha=0.75)
plt.title('SVM Accuracy Comparison for Different Kernels')
plt.ylabel('Accuracy (in percentage)')
plt.ylim(0, 100)
plt.axhline(y=50, color='r', linestyle='--')  
plt.show()
