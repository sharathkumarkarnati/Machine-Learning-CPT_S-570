from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load training data and labels
with open("/Users/sharathkarnati/Desktop/ML/KARNATI-011852253/Assignment 3/data/traindata.txt", "r") as file:
    train_documents = file.read().splitlines()

with open("/Users/sharathkarnati/Desktop/ML/KARNATI-011852253/Assignment 3/data/trainlabels.txt", "r") as file:
    train_labels = [int(label) for label in file.read().splitlines()]

# Load testing data and labels
with open("/Users/sharathkarnati/Desktop/ML/KARNATI-011852253/Assignment 3/data/testdata.txt", "r") as file:
    test_documents = file.read().splitlines()

with open("/Users/sharathkarnati/Desktop/ML/KARNATI-011852253/Assignment 3/data/testlabels.txt", "r") as file:
    test_labels = [int(label) for label in file.read().splitlines()]

# Load stopwords
stopwords = set()
with open("/Users/sharathkarnati/Desktop/ML/KARNATI-011852253/Assignment 3/data/stoplist.txt", "r") as file:
    stopwords = {line.strip() for line in file}

# Clean the training data by removing stopwords
filtered_train_documents = []
for doc in train_documents:
    words = doc.split()
    filtered_words = [word for word in words if word.lower() not in stopwords]
    filtered_train_documents.append(" ".join(filtered_words))

# Initialize CountVectorizer and transform documents to feature vectors
vectorizer = CountVectorizer()
X_train_features = vectorizer.fit_transform(filtered_train_documents)
X_test_features = vectorizer.transform(test_documents)

# Initialize and train the Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_features, train_labels)

# Make predictions on the training set
train_predictions = nb_classifier.predict(X_train_features)
train_accuracy = accuracy_score(train_labels, train_predictions)
print(f"Training Accuracy: {train_accuracy}")

# Make predictions on the test set
test_predictions = nb_classifier.predict(X_test_features)
test_accuracy = accuracy_score(test_labels, test_predictions)
print(f"Testing Accuracy: {test_accuracy}")

# Generate classification report for detailed performance metrics
report = classification_report(test_labels, test_predictions)
print(report)
