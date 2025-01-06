import numpy as np
from collections import Counter

class TextVectorizer:
    def __init__(self):
        self.word_index = {}

    def fit_and_transform(self, documents):
        self.word_index = self._create_vocab(documents)
        return self._encode_documents(documents)

    def transform(self, documents):
        return self._encode_documents(documents)

    def _create_vocab(self, documents):
        vocabulary = {}
        for doc in documents:
            words = doc.split()
            for word in words:
                if word not in vocabulary:
                    vocabulary[word] = len(vocabulary)
        return vocabulary

    def _encode_documents(self, documents):
        num_docs = len(documents)
        num_unique_words = len(self.word_index)
        matrix = np.zeros((num_docs, num_unique_words), dtype=int)

        for i, doc in enumerate(documents):
            word_count = Counter(doc.split())
            for word, count in word_count.items():
                if word in self.word_index:
                    index = self.word_index[word]
                    matrix[i, index] = count

        return matrix


class LaplaceNaiveBayes:
    def __init__(self, smoothing_param=1.0):
        self.alpha = smoothing_param  # Laplace smoothing
        self.class_probabilities = {}
        self.word_given_class_probabilities = {}

    def fit(self, feature_matrix, labels):
        num_docs, num_features = feature_matrix.shape
        self.unique_classes = np.unique(labels)

        for cls in self.unique_classes:
            class_mask = labels == cls
            class_doc_count = np.sum(class_mask)
            self.class_probabilities[cls] = (class_doc_count + self.alpha) / (
                num_docs + len(self.unique_classes) * self.alpha
            )

            word_counts_in_class = feature_matrix[class_mask].sum(axis=0)
            total_word_counts = feature_matrix.sum(axis=0)
            self.word_given_class_probabilities[cls] = (word_counts_in_class + self.alpha) / (
                total_word_counts + num_features * self.alpha
            )

    def predict(self, feature_matrix):
        predictions = []
        for doc in feature_matrix:
            class_scores = {}
            for cls in self.unique_classes:
                class_scores[cls] = np.log(self.class_probabilities[cls])
                for word_idx, word_count in enumerate(doc):
                    if word_count > 0:
                        class_scores[cls] += np.log(self.word_given_class_probabilities[cls][word_idx])
            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)
        return predictions


# Load stop words from file
stopwords = set()
with open("/Users/sharathkarnati/Desktop/ML/KARNATI-011852253/Assignment 3/data/stoplist.txt", "r") as file:
    stopwords = {line.strip() for line in file}

# Load training and testing data and labels
with open("/Users/sharathkarnati/Desktop/ML/KARNATI-011852253/Assignment 3/data/traindata.txt", "r") as file:
    train_docs = file.read().splitlines()

with open("/Users/sharathkarnati/Desktop/ML/KARNATI-011852253/Assignment 3/data/trainlabels.txt", "r") as file:
    train_labels = list(map(int, file.read().splitlines()))

with open("/Users/sharathkarnati/Desktop/ML/KARNATI-011852253/Assignment 3/data/testdata.txt", "r") as file:
    test_docs = file.read().splitlines()

with open("/Users/sharathkarnati/Desktop/ML/KARNATI-011852253/Assignment 3/data/testlabels.txt", "r") as file:
    test_labels = list(map(int, file.read().splitlines()))

# Clean the training data by removing stopwords
cleaned_train_docs = []
for doc in train_docs:
    filtered_words = [word for word in doc.split() if word.lower() not in stopwords]
    cleaned_train_docs.append(" ".join(filtered_words))

# Vectorize the documents
vectorizer = TextVectorizer()
X_train = vectorizer.fit_and_transform(cleaned_train_docs)
X_test = vectorizer.transform(test_docs)

# Train and evaluate Naive Bayes model
nb_model = LaplaceNaiveBayes()
nb_model.fit(X_train, train_labels)

train_preds = nb_model.predict(X_train)
test_preds = nb_model.predict(X_test)

# Calculate accuracy
from sklearn.metrics import accuracy_score

train_accuracy = accuracy_score(train_labels, train_preds)
print(f"Training Accuracy: {train_accuracy}")

test_accuracy = accuracy_score(test_labels, test_preds)
print(f"Test Accuracy: {test_accuracy}")

# Save results to file
with open("/Users/sharathkarnati/Desktop/ML/KARNATI-011852253/Assignment 3/output.txt", "w") as output_file:
    output_file.write(f"Training Accuracy: {train_accuracy}\n")
    output_file.write(f"Testing Accuracy: {test_accuracy}\n")
