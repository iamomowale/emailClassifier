import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split

from keras.models import Sequential
from keras.layers import (
    Embedding, LSTM, GRU, Dense, Dropout, Bidirectional
)
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder, StandardScaler
from ANN_CNN_RNN_model import showMetrics, TimeHistory
from statistics import mean


def LSTMmodel(data, classLabel):
    X = data['emails']
    Y = data[classLabel]

    # Encode the labels (assuming binary classification, you can modify for multi-class)
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)

    # Split data into training and test sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

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

    # Add LSTM layer
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.5))  # Add dropout for regularization
    model.add(LSTM(32))

    # Output layer with 2 nodes (binary classification: spam and ham)
    model.add(Dense(2, activation='softmax'))

    # Compile the classifier model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=3e-4), metrics=['accuracy'])

    # Fit the model on the training set
    model.fit(X_train_padded, y_train, epochs=20, batch_size=32, validation_split=0.1,callbacks=[time_callback]) #20 epochs

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(X_test_padded, y_test)
    print("Test accuracy:", test_acc)
    print("Loss:", test_loss)
    avgEpochTime = mean(time_callback.times)
    print("Avg Epoch Time: ", avgEpochTime)

    showMetrics(model, X_test_padded, y_test)


def GRUmodel(data, classLabel):
    X = data['emails']
    Y = data[classLabel]

    # Encode the labels (assuming binary classification, you can modify for multi-class)
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)

    # Split data into training and test sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # Tokenize and pad the email text
    max_words = 10000  # Maximum number of unique words in your vocabulary
    max_len = 200  # Maximum length of sequences
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_len)
    X_test_padded = pad_sequences(X_test_seq, maxlen=max_len)

    # Initialize the GRU model
    model = Sequential()
    time_callback = TimeHistory()

    # Add an embedding layer
    embedding_dim = 128
    model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len))

    # Add GRU layer
    model.add(GRU(64, return_sequences=True))
    model.add(Dropout(0.5))  # Add dropout for regularization
    model.add(GRU(32))

    # Output layer with 2 nodes (binary classification: spam and ham)
    model.add(Dense(2, activation='softmax'))

    # Compile the classifier model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=3e-4), metrics=['accuracy'])

    # Fit the model on the training set
    model.fit(X_train_padded, y_train, epochs=30, batch_size=32, validation_split=0.1, callbacks=[time_callback]) #30 epochs

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(X_test_padded, y_test)
    print("Test accuracy:", test_acc)
    avgEpochTime = mean(time_callback.times)
    print("Avg Epoch Time: ", avgEpochTime)

    showMetrics(model, X_test_padded, y_test)


def LSTMspamBase(data):
    # Assuming your DataFrame has features in columns 0-56 and the label in column 57
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values

    # Split the dataset into training and test sets (70% train, 30% test)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # Standardize the data for feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Initialize the LSTM model
    model = Sequential()
    time_callback = TimeHistory()

    # Add an embedding layer
    embedding_dim = 128
    model.add(Embedding(input_dim=X_train.shape[1], output_dim=embedding_dim, input_length=X_train.shape[1]))

    # Add LSTM layer
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.5))  # Add dropout for regularization
    model.add(LSTM(32))

    # Output layer with 2 nodes (binary classification: spam and not spam)
    model.add(Dense(2, activation='softmax'))

    # Compile the classifier model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=3e-4), metrics=['accuracy'])

    # Fit the model on the training set
    model.fit(X_train, Y_train, epochs=50, batch_size=16, callbacks=[time_callback])

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(X_test, Y_test)
    print("Test accuracy:", test_acc)
    print("Loss:", test_loss)
    avgEpochTime = mean(time_callback.times)
    print("Avg Epoch Time: ", avgEpochTime)

    showMetrics(model, X_test, Y_test)


def GRUspamBase(data):
    # Assuming your DataFrame has features in columns 0-56 and the label in column 57
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values

    # Split the dataset into training and test sets (70% train, 30% test)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # Standardize the data for feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Initialize the GRU model
    model = Sequential()
    time_callback = TimeHistory()

    # Add an embedding layer
    embedding_dim = 128
    model.add(Embedding(input_dim=X_train.shape[1], output_dim=embedding_dim, input_length=X_train.shape[1]))

    # Add GRU layer
    model.add(GRU(64, return_sequences=True))
    model.add(Dropout(0.5))  # Add dropout for regularization
    model.add(GRU(32))

    # Output layer with 2 nodes (binary classification: spam and not spam)
    model.add(Dense(2, activation='softmax'))

    # Compile the classifier model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=3e-4), metrics=['accuracy'])

    # Fit the model on the training set
    model.fit(X_train, Y_train, epochs=30, batch_size=16, callbacks=[time_callback])

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(X_test, Y_test)
    print("Test accuracy:", test_acc)
    print("Loss:", test_loss)
    avgEpochTime = mean(time_callback.times)
    print("Avg Epoch Time: ", avgEpochTime)


    # Make predictions
    Y_pred = model.predict(X_test)
    Y_pred_classes = np.argmax(Y_pred, axis=1)

    showMetrics(model,X_test, Y_test)

def BiLSTMmodel(data, classLabel):
    X = data['emails']
    Y = data[classLabel]

    # Encode the labels (assuming binary classification, you can modify for multi-class)
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)

    # Split data into training and test sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    # Tokenize and pad the email text
    max_words = 10000  # Maximum number of unique words in your vocabulary
    max_len = 200  # Maximum length of sequences
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_len)
    X_test_padded = pad_sequences(X_test_seq, maxlen=max_len)

    # Initialize the Bi-LSTM model
    model = Sequential()
    time_callback = TimeHistory()

    # Add an embedding layer
    embedding_dim = 128
    model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len))

    # Add Bidirectional LSTM layer
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.5))  # Add dropout for regularization
    model.add(Bidirectional(LSTM(32)))

    # Output layer with 2 nodes (binary classification: spam and ham)
    model.add(Dense(2, activation='softmax'))

    # Compile the classifier model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=3e-4), metrics=['accuracy'])

    # Fit the model on the training set
    model.fit(X_train_padded, y_train, epochs=20, batch_size=32, validation_split=0.1, callbacks=[time_callback]) #20 epochs

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(X_test_padded, y_test)
    print("Test accuracy:", test_acc)
    print("Loss:", test_loss)
    avgEpochTime = mean(time_callback.times)
    print("Avg Epoch Time: ", avgEpochTime)
    showMetrics(model, X_test_padded, y_test)

def BiLSTMspamBase(data):
    # Assuming your DataFrame has features in columns 0-56 and the label in column 57
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values

    # Split the dataset into training and test sets (70% train, 30% test)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # Standardize the data for feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Initialize the LSTM model
    model = Sequential()
    time_callback = TimeHistory()

    # Add an embedding layer
    embedding_dim = 128
    model.add(Embedding(input_dim=X_train.shape[1], output_dim=embedding_dim, input_length=X_train.shape[1]))

    # Add Bidirectional LSTM layer
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.5))  # Add dropout for regularization
    model.add(Bidirectional(LSTM(32)))

    # Output layer with 2 nodes (binary classification: spam and ham)
    model.add(Dense(2, activation='softmax'))

    # Compile the classifier model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

    # Fit the model on the training set
    model.fit(X_train, Y_train, epochs=10, batch_size=32, callbacks=[time_callback])


    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(X_test, Y_test)
    print("Test accuracy:", test_acc)
    print("Loss:", test_loss)
    avgEpochTime = mean(time_callback.times)
    print("Avg Epoch Time: ", avgEpochTime)

    showMetrics(model, X_test, Y_test)
