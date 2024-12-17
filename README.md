
# Spam Detection Using FFNN and Naive Bayes

We are using a email classfication dataset with 3002 Column and 1000 rows of data.
We are using Naive Byse Model and FFNN to train a model to detect is the Email is SPAM or not. 

Mission of this project is to learn and train a model to help people to detect the SPAM Mail. So they can safe out from virus and malware and other spam thing.


## Author

- [@rahatTamzid](https://www.github.com/rahatTamzid)


## Dataset

[Download Dataset](https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv)

## Documentation

In Both Model We have use the 80% of the data from dataset. and 20% of data for testing.


## Naive bayes
* Setup the dataset in the directory of the code file.
```python
dataset = pd.read_csv('./emails.csv')
```

you can change the dataset from here.

* I am assigning 2 variable to design the dataset | Main_dataset variable I am dropping 1st and last column. Main_predication_dataset varible the predication.

```python
Main_dataset = dataset.drop(columns=["Email No.", "Prediction"])
Main_predication_dataset = dataset["Prediction"]
```

* Training the Model
```python
Main_Trained, Trained_tested, Predication_train, Predication_test = train_test_split(train_set, predication_set, test_size=0.1, random_state=150)

classifier = MultinomialNB()
classifier.fit(Main_Trained, Predication_train)

Model_Predication = classifier.predict(Trained_tested)
```

* Getting F1 Score, Config Martix and classification Report form the Model
```python
conf_matrix = confusion_matrix(Predication_test, Model_Predication)
f1 = f1_score(Predication_test, Model_Predication)
class_report = classification_report(Predication_test, Model_Predication)
```

* So as we know naive byse train model return result for one data at a time. let's keep the data in a array and show the plot
```python

train_sizes = np.linspace(0.1, 0.99, 10)
f1_scores = []
accuracies = []

for train_size in train_sizes:
    X_train_partial, _, y_train_partial, _ = train_test_split(
        Main_Trained, Predication_train, train_size=train_size, random_state=150
    )

    classifier.fit(X_train_partial, y_train_partial)
    y_pred = classifier.predict(Trained_tested)

    f1_scores.append(f1_score(Predication_test, y_pred))
    accuracies.append(accuracy_score(Predication_test, y_pred))
```

* Showing plot
```python
print(f1_scores)
print(accuracies)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, f1_scores, label='F1 Score', marker='o')
plt.plot(train_sizes, accuracies, label='Accuracy', marker='o')
plt.xlabel('Training Set Size (Proportion)')
plt.ylabel('Performance')
plt.title('Naive Bayes Performance Over Increasing Training Set Size')
plt.legend()
plt.grid(True)
plt.show()
```

![ScreenShot](/screenshot/bayes.png)





## FFNN

* Setup the dataset in the directory of the code file.
```python
dataset = pd.read_csv('./emails.csv')
```

you can change the dataset from here.

* I am assigning 2 variable to design the dataset | Main_dataset variable I am dropping 1st and last column. Main_predication_dataset varible the predication.

```python
Main_dataset = dataset.drop(columns=["Email No.", "Prediction"])
Main_predication_dataset = dataset["Prediction"]
```

* Traning the Model and Using 100 Epochs and Using 30 batch Size.
```python
main_dataset_train, main_dataset_test, main_predication_train, main_predication_test = train_test_split(Main_dataset, Main_predication_dataset, test_size=0.2, random_state=56)

scaler = StandardScaler()
Main_train = scaler.fit_transform(main_dataset_train)
Main_test = scaler.transform(main_dataset_test)

model = Sequential([
    Dense(128, activation='relu', input_shape=(Main_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])


model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(Main_train, main_predication_train, epochs=100, batch_size=30,
                   validation_data=(Main_test, main_predication_test), verbose=1)

main_trained = (model.predict(Main_test) > 0.5).astype(int)
```

![ScreenShot](/screenshot/ffnn.png)
