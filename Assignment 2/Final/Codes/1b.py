import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


train_data = pd.read_csv("/Users/sharathkarnati/Desktop/ML/KARNATI-011852253/Assignment 2/data/fashion-mnist_train.csv")
test_data = pd.read_csv("/Users/sharathkarnati/Desktop/ML/KARNATI-011852253/Assignment 2/data/fashion-mnist_test.csv")

train_X = train_data.iloc[:, 1:].values
train_Y = train_data.iloc[:, 0].values
test_X = test_data.iloc[:, 1:].values
test_Y = test_data.iloc[:, 0].values


X_train, X_val, y_train, y_val = train_test_split(train_X, train_Y, test_size=0.2, random_state=42, shuffle=True)


C_values = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
val_accuracy = []


for C in C_values:
    svm_classifier = SVC(kernel='linear', C=C, max_iter=3000)
    svm_classifier.fit(X_train, y_train)
    svm_val_predictions = svm_classifier.predict(X_val)
    accuracy = metrics.accuracy_score(y_val, svm_val_predictions)
    val_accuracy.append(accuracy)
    print(f'C={C}, Validation Accuracy: {accuracy * 100:.2f}%')


best_C = C_values[np.argmax(val_accuracy)]
print(f'Best C value based on validation accuracy: {best_C}')


final_X = np.vstack((X_train, X_val))
final_y = np.concatenate((y_train, y_val))

final_svm_classifier = SVC(kernel='linear', C=best_C, max_iter=3000)
final_svm_classifier.fit(final_X, final_y)


svm_test_predictions = final_svm_classifier.predict(test_X)
test_accuracy = metrics.accuracy_score(test_Y, svm_test_predictions)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

conf_mat = confusion_matrix(test_Y, svm_test_predictions)


conf_table = pd.DataFrame(conf_mat,
                           columns=["Tshirt/Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker",
                                    "Bag", "Ankle-Boot"],
                           index=["Tshirt/Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker",
                                  "Bag", "Ankle-Boot"])


plt.figure(figsize=(10, 7))
heat_map = sns.heatmap(conf_table, annot=True, fmt="d", cmap="OrRd")
heat_map.set_title('Confusion Matrix', size=16)
heat_map.set_xlabel('Predicted Labels', size=14)
heat_map.set_ylabel('True Labels', size=14)
plt.savefig('/Users/sharathkarnati/Desktop/ML/KARNATI-011852253/Assignment 2/output/1.b_Confusion_matrix.png')
plt.show()
