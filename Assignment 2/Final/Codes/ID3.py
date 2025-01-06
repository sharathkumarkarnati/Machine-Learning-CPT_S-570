import numpy as np

# Entropy calculation
def entropy(y):
    class_counts = np.bincount(y)
    probabilities = class_counts / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

# Subset function that splits the data based on a threshold
def subset(X, y, feature_index, threshold):
    X_left = X[X[:, feature_index] < threshold]
    y_left = y[X[:, feature_index] < threshold]
    X_right = X[X[:, feature_index] >= threshold]
    y_right = y[X[:, feature_index] >= threshold]
    return X_left, y_left, X_right, y_right

# Function to compute the best feature and threshold to split on
def best_split(X, y):
    n_features = X.shape[1]
    base_entropy = entropy(y)
    best_info_gain = -1
    best_feature_index = None
    best_threshold = None
    
    for feature_index in range(n_features):
        sorted_feature_values = np.sort(X[:, feature_index])
        thresholds = (sorted_feature_values[:-1] + sorted_feature_values[1:]) / 2
        
        for threshold in thresholds:
            X_left, y_left, X_right, y_right = subset(X, y, feature_index, threshold)
            if len(y_left) == 0 or len(y_right) == 0:
                continue
                
            entropy_left = entropy(y_left)
            entropy_right = entropy(y_right)
            weighted_entropy = (len(y_left) / len(y) * entropy_left) + (len(y_right) / len(y) * entropy_right)
            info_gain = base_entropy - weighted_entropy
            
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature_index = feature_index
                best_threshold = threshold
                
    return best_feature_index, best_threshold

# Recursive function to build the decision tree
def build_tree(X, y, max_depth=None, depth=0):
    if len(np.unique(y)) == 1:
        return np.unique(y)[0]
    
    if max_depth is not None and depth >= max_depth:
        return np.bincount(y).argmax()
    
    feature_index, threshold = best_split(X, y)
    
    if feature_index is None:
        return np.bincount(y).argmax()
    
    X_left, y_left, X_right, y_right = subset(X, y, feature_index, threshold)
    
    if len(y_left) == 0 or len(y_right) == 0:
        return np.bincount(y).argmax()
    
    left_subtree = build_tree(X_left, y_left, max_depth, depth + 1)
    right_subtree = build_tree(X_right, y_right, max_depth, depth + 1)
    
    return {"feature_index": feature_index, "threshold": threshold, "left": left_subtree, "right": right_subtree}

# Prediction function
def predict(tree, X_test):
    if isinstance(tree, dict):
        feature_index = tree['feature_index']
        threshold = tree['threshold']
        if X_test[feature_index] < threshold:
            return predict(tree['left'], X_test)
        else:
            return predict(tree['right'], X_test)
    else:
        return tree

# Pruning the tree 
def prune(tree, X_val, y_val):
    if isinstance(tree, dict):
        tree['left'] = prune(tree['left'], X_val, y_val)
        tree['right'] = prune(tree['right'], X_val, y_val)
        
        if not isinstance(tree['left'], dict) and not isinstance(tree['right'], dict):
            left_leaf = tree['left']
            right_leaf = tree['right']
            majority_class = np.bincount([left_leaf, right_leaf]).argmax()
            
            pruned_tree = majority_class
            accuracy_before_pruning = np.mean([predict(tree, x) == y for x, y in zip(X_val, y_val)])
            accuracy_after_pruning = np.mean([predict(pruned_tree, x) == y for x, y in zip(X_val, y_val)])
            
            if accuracy_after_pruning >= accuracy_before_pruning:
                return pruned_tree
        return tree
    return tree
