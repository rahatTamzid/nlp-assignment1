import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
import matplotlib.pyplot as plt




dataset = pd.read_csv('./emails.csv')
train_set = dataset.drop(columns=["Email No.", "Prediction"])
predication_set = dataset['Prediction']

Main_Trained, Trained_tested, Predication_train, Predication_test = train_test_split(train_set, predication_set, test_size=0.1, random_state=150)


# Train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(Main_Trained, Predication_train)

# Make predictions on the test set
Model_Predication = classifier.predict(Trained_tested)

# Calculate confusion matrix, F1 score, and classification report
conf_matrix = confusion_matrix(Predication_test, Model_Predication)
f1 = f1_score(Predication_test, Model_Predication)
class_report = classification_report(Predication_test, Model_Predication)

train_sizes = np.linspace(0.1, 0.99, 10)
f1_scores = []
accuracies = []


# Loop through different training set sizes to simulate epochs
for train_size in train_sizes:
    X_train_partial, _, y_train_partial, _ = train_test_split(
        Main_Trained, Predication_train, train_size=train_size, random_state=150
    )
    
    classifier.fit(X_train_partial, y_train_partial)
    y_pred = classifier.predict(Trained_tested)
    
    f1_scores.append(f1_score(Predication_test, y_pred))
    accuracies.append(accuracy_score(Predication_test, y_pred))

# Plot the performance metrics
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, f1_scores, label='F1 Score', marker='o')
plt.plot(train_sizes, accuracies, label='Accuracy', marker='o')
plt.xlabel('Training Set Size (Proportion)')
plt.ylabel('Performance')
plt.title('Naive Bayes Performance Over Increasing Training Set Size')
plt.legend()
plt.grid(True)
plt.show()