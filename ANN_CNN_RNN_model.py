import numpy as np
import os.path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Embedding
from keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn import metrics
from sklearn.metrics import (
accuracy_score, precision_score, recall_score, f1_score,
cohen_kappa_score, classification_report, confusion_matrix,matthews_corrcoef, roc_auc_score,
)
import matplotlib.pyplot as plt
from keras.layers import (
    Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Embedding, Input, GlobalMaxPooling1D
)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import time
from statistics import mean
from numpy import mean


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

def annModel(data, classLabel):

    X = data['emails']  # Assuming 'emails' is the column with tokenized and lemmatized email text
    y = data[classLabel]  # Assuming 'labels' is the column with labels

    # Split data into training and test sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # TF-IDF Vectorization with English stop words and Chi-Square feature selection
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    chi2_selector = SelectKBest(chi2, k=300)
    X_train_selected = chi2_selector.fit_transform(X_train_tfidf, y_train)
    X_test_selected = chi2_selector.transform(X_test_tfidf)

    # Convert SparseTensor to dense NumPy arrays
    X_train_selected = X_train_selected.toarray()
    X_test_selected = X_test_selected.toarray()

    # Initialize the sequential deep learning model
    model = Sequential()
    time_callback = TimeHistory()

    # Input layer with 40 neurons
    model.add(Dense(units=40, input_dim=300, activation='relu'))

    # First hidden layer with 32 neurons and ReLU activation
    model.add(Dense(units=32, activation='relu'))

    # Second hidden layer with 16 neurons and ReLU activation
    model.add(Dense(units=16, activation='relu'))

    # Output layer with 2 nodes (spam and ham) and softmax activation
    model.add(Dense(units=2, activation='softmax'))

    # Compile the classifier model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

    # Fit the model on the training set
    model.fit(X_train_selected, y_train, epochs=200, batch_size=16, callbacks=[time_callback]) #was 200 epochs

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(X_test_selected, y_test)
    print("Test accuracy:", test_acc)
    avgEpochTime = mean(time_callback.times)
    print("Avg Epoch Time: ", avgEpochTime)


    avgEpochTime = mean(time_callback.times)
    print("Avg Epoch Time: ", avgEpochTime)

    showMetrics(model, X_test_selected, y_test)

    # Return the email classifier
    # You can load the model later using: loaded_model = keras.models.load_model('email_classifier.h5')

def ANNspamBase(data):
    X = data.iloc[:,0:-1].values  # All columns excepts the last one for labels
    y = data.iloc[:,-1].values  #The last column with labels

    # Split data into training and test sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    # Standardize the test for feature scalling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Initialize the sequential deep learning model
    model = Sequential()
    time_callback = TimeHistory()

    model.add(Dense(units=40, input_dim=57, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=2, activation='softmax'))

    # Compile the classifier model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

    # Fit the model on the training set
    model.fit(X_train, y_train, epochs=100, batch_size=16, callbacks=[time_callback])

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("Test accuracy:", test_acc)
    avgEpochTime = mean(time_callback.times)
    print("Avg Epoch Time: ", avgEpochTime)

    showMetrics(model, X_test, y_test)

    # Return the email classifier
    # You can load the model later using: loaded_model = keras.models.load_model('email_classifier.h5')"""

def showMetrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Calculate various classification metrics
    accuracy = accuracy_score(y_test, y_pred_classes)
    precision = precision_score(y_test, y_pred_classes)
    recall = recall_score(y_test, y_pred_classes)
    f1 = f1_score(y_test, y_pred_classes)
    kappa = cohen_kappa_score(y_test, y_pred_classes)
    confusion = confusion_matrix(y_test, y_pred_classes)
    mcc = matthews_corrcoef(y_test, y_pred_classes)

    #Calculate the AUC score
    y_pred_proba = model.predict(X_test)
    y_true = np.array(y_test)
    auc_roc = roc_auc_score(y_true, y_pred_proba[:,1])

    # Specificity (True Negative Rate)
    tn, fp, fn, tp = confusion.ravel()
    specificity = tn / (tn + fp)

    report = classification_report(y_test, y_pred_classes)

    # Print and display the results
    print("Accuracy: {:.4f}".format(accuracy))
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1-Score: {:.4f}".format(f1))
    print("Cohen's Kappa: {:.4f}".format(kappa))
    print(f"Specificity: {specificity:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print("\nClassification Report:")
    print(report)

    # Plotting Confusion Matrix
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=[True, False])
    cm_display.plot()
    plt.show()

def cnnModel(data, classLabel):
    X = data['emails']  # Assuming 'emails' is the column with tokenized and lemmatized email text
    y = data[classLabel]  # Assuming 'labels' is the column with labels

    # Split data into training and test sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Tokenize and pad the email text
    max_words = 10000  # Maximum number of unique words in your vocabulary
    max_len = 200  # Maximum length of sequences
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_len)
    X_test_padded = pad_sequences(X_test_seq, maxlen=max_len)

    # Initialize the CNN model
    model = Sequential()
    time_callback = TimeHistory()

    # Add an embedding layer
    embedding_dim = 128
    model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len))

    # Add 1D Convolutional layer
    filters = 64
    kernel_size = 3
    model.add(Conv1D(filters, kernel_size, activation='relu'))

    # Add Global Max Pooling layer
    model.add(GlobalMaxPooling1D())

    # Add two hidden layers
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))  # Add dropout for regularization
    model.add(Dense(32, activation='relu'))

    # Output layer with 2 nodes (binary classification: spam and ham)
    model.add(Dense(2, activation='softmax'))

    # Compile the classifier model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model on the training set
    model.fit(X_train_padded, y_train, epochs=100, batch_size=16, validation_split=0.1, callbacks=[time_callback]) #100 epochs

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(X_test_padded, y_test)
    print("Test accuracy:", test_acc)
    avgEpochTime = mean(time_callback.times)
    print("Avg Epoch Time: ", avgEpochTime)

    showMetrics(model, X_test_padded, y_test)

def CNNspamBase(data):
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Reshape the data for compatibility with the CNN model
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    model = Sequential()
    time_callback = TimeHistory()
    # Add convolutional layers
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Flatten())

    # Add fully connected layers
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Fit the model on the training set
    model.fit(X_train, Y_train, epochs=100, batch_size=16, callbacks=[time_callback])

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(X_test, Y_test)
    print("Test accuracy:", test_acc)

    avgEpochTime = mean(time_callback.times)
    print("Avg Epoch Time: ", avgEpochTime)

    showMetrics(model, X_test, Y_test)

def rnnModel(data, classLabel):
    X = data['emails']  # Assuming 'emails' is the column with tokenized and lemmatized email text
    y = data[classLabel]  # Assuming 'labels' is the column with labels

    # Split data into training and test sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # TF-IDF Vectorization with English stop words and Chi-Square feature selection
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    chi2_selector = SelectKBest(chi2, k=300)
    X_train_selected = chi2_selector.fit_transform(X_train_tfidf, y_train)
    X_test_selected = chi2_selector.transform(X_test_tfidf)

    # Convert SparseTensor to dense NumPy arrays
    X_train_selected = X_train_selected.toarray()
    X_test_selected = X_test_selected.toarray()

    # Reshape data for SimpleRNN input (assuming time steps = 1)
    X_train_selected = X_train_selected.reshape((X_train_selected.shape[0], 1, X_train_selected.shape[1]))
    X_test_selected = X_test_selected.reshape((X_test_selected.shape[0], 1, X_test_selected.shape[1]))

    # Initialize the sequential deep learning model
    model = Sequential()
    time_callback = TimeHistory()

    # Add an Embedding layer for sequence input
    model.add(SimpleRNN(128, input_shape=(X_train_selected.shape[1], X_train_selected.shape[2])))

    # Dense layers (you can modify this based on your requirements)
    model.add(Dense(units=40, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=16, activation='relu'))

    # Output layer with 2 nodes (spam and ham) and softmax activation
    model.add(Dense(units=2, activation='softmax'))

    # Compile the classifier model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

    # Fit the model on the training set
    model.fit(X_train_selected, y_train, epochs=20, batch_size=16, callbacks=[time_callback]) #20 epochs

    # Evaluate the model on the test set
    avgEpochTime = mean(time_callback.times)
    print("Avg Epoch Time: ", avgEpochTime)
    showMetrics(model, X_test_selected, y_test)


def rnnSpambase(data):
    X = data.iloc[:, 0:-1].values  # All columns excepts the last one for labels
    y = data.iloc[:, -1].values  # The last column with labels

    # Split data into training and test sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    # Standardize the test for feature scalling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Reshape data for SimpleRNN input (assuming time steps = number of features)
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Initialize the sequential deep learning model
    model = Sequential()
    time_callback = TimeHistory()

    # Add an Embedding layer for sequence input
    model.add(SimpleRNN(128, input_shape=(X_train.shape[1], X_train.shape[2])))

    # Dense layers (you can modify this based on your requirements)
    model.add(Dense(units=40, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=16, activation='relu'))

    # Output layer with 2 nodes (spam and ham) and softmax activation
    model.add(Dense(units=2, activation='softmax'))

    # Compile the classifier model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

    # Fit the model on the training set
    model.fit(X_train, y_train, epochs=20, batch_size=16, callbacks=[time_callback])

    # Evaluate the model on the test set
    avgEpochTime = mean(time_callback.times)
    print("Avg Epoch Time: ", avgEpochTime)
    showMetrics(model,X_test, y_test)