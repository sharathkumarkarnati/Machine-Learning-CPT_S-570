import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import ID3 

# Load dataset
def load_breast_cancer_data():
    data = pd.read_csv("/Users/sharathkarnati/Desktop/ML/KARNATI-011852253/Assignment 2/data/breast+cancer+wisconsin+diagnostic/wdbc.data", header=None)
    X = data.iloc[:, 2:].values
    y = np.where(data.iloc[:, 1] == 'M', 1, 0)
    return X, y

# Main execution
if __name__ == "__main__":
  
    X, y = load_breast_cancer_data()
    
   
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.66, random_state=42)

    # Part (b): Train the decision tree
    tree = ID3.build_tree(X_train, y_train, max_depth=5)
    
   
    y_val_pred = [ID3.predict(tree, x) for x in X_val]
    y_test_pred = [ID3.predict(tree, x) for x in X_test]
    
    val_accuracy = accuracy_score(y_val, y_val_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Validation Accuracy (without pruning): {val_accuracy:.2f}")
    print(f"Test Accuracy (without pruning): {test_accuracy:.2f}")
    
    # Part (d): Prune the tree using validation set
    pruned_tree = ID3.prune(tree, X_val, y_val)

    # Compute validation and test accuracy after pruning
    y_val_pred_pruned = [ID3.predict(pruned_tree, x) for x in X_val]
    y_test_pred_pruned = [ID3.predict(pruned_tree, x) for x in X_test]

    val_accuracy_pruned = accuracy_score(y_val, y_val_pred_pruned)
    test_accuracy_pruned = accuracy_score(y_test, y_test_pred_pruned)

    print(f"Validation Accuracy (after pruning): {val_accuracy_pruned:.2f}")
    print(f"Test Accuracy (after pruning): {test_accuracy_pruned:.2f}")

    # Observations
    print("\nObservations:")
    if val_accuracy_pruned > val_accuracy:
        print("Pruning improved the validation accuracy.")
    else:
        print("Pruning did not improve the validation accuracy.")

    if test_accuracy_pruned > test_accuracy:
        print("Pruning improved the test accuracy.")
    else:
        print("Pruning did not improve the test accuracy.")
