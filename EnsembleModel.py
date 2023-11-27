
from keras.optimizers import Adam
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
accuracy_score, precision_score, recall_score, f1_score,
cohen_kappa_score, classification_report, confusion_matrix,matthews_corrcoef, roc_auc_score,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from scikeras.wrappers import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPooling1D, GlobalMaxPooling1D, \
    Dropout, SimpleRNN, GRU, Bidirectional, Flatten
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from ANN_CNN_RNN_model import showMetrics

def ensembleModels(data, classLabel, ensembleType):

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data['emails'], data[classLabel], test_size=0.2, random_state=42)

    # Tokenize and pad the email text
    max_words = 10000
    max_len = 200
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_len)
    X_test_padded = pad_sequences(X_test_seq, maxlen=max_len)

    # Define your deep learning models
    def create_ann_model():
        model = Sequential()
        model.add(Dense(units=40, input_dim=max_len, activation='relu'))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=16, activation='relu'))
        model.add(Dense(units=2, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
        return model

    def create_cnn_model():
        model = Sequential()
        model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_len))
        filters = 64
        kernel_size = 3
        model.add(Conv1D(filters, kernel_size, activation='relu'))

        model.add(GlobalMaxPooling1D())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))  # Add dropout for regularization
        model.add(Dense(32, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def create_lstm_model():
        model = Sequential()
        model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_len))
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.5))  # Add dropout for regularization
        model.add(LSTM(32))
        model.add(Dense(2, activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=3e-4), metrics=['accuracy'])
        return model

    def create_gru_model():
        model = Sequential()
        embedding_dim = 128
        model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len))
        model.add(GRU(64, return_sequences=True))
        model.add(Dropout(0.5))  # Add dropout for regularization
        model.add(GRU(32))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=3e-4), metrics=['accuracy'])
        return model
    def create_biLSTM_model():
        model = Sequential()
        embedding_dim = 128
        model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len))

        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.5))  # Add dropout for regularization
        model.add(Bidirectional(LSTM(32)))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=3e-4), metrics=['accuracy'])
        return model

    # Create instances of your deep learning models using KerasClassifier wrapper
    ann_model = KerasClassifier(model=create_ann_model, epochs=10, batch_size=16, verbose=0)
    cnn_model = KerasClassifier(model=create_cnn_model, epochs=5, batch_size=16, verbose=0)
    lstm_model = KerasClassifier(model=create_lstm_model, epochs=5, batch_size=32, verbose=0)
    #rnn_model = KerasClassifier(model=create_rnn_model, epochs=5, batch_size=16, verbose=0)
    gru_model = KerasClassifier(model=create_gru_model, epochs=5, batch_size=32, verbose=0)
    biLSM_model = KerasClassifier(model=create_biLSTM_model, epochs=5, batch_size=32, verbose=0)

    if ensembleType == "stacking":
        # Create a stacking ensemble with MLP (ANN), Random Forest, and SVM as meta-classifier
        base_classifiers = [
            ('ann', ann_model),
            ('cnn', cnn_model),
            ('lstm', lstm_model),
            #('rnn', rnn_model),
            ('gru', gru_model),
            ('bilstm', biLSM_model)
        ]

        stacking_classifier = StackingClassifier(
            estimators=base_classifiers,
            final_estimator=RandomForestClassifier(n_estimators=50),
            stack_method='predict_proba'  # Use 'predict_proba' to get class probabilities for the meta-classifier
        )

        # Train the stacking ensemble
        stacking_classifier.fit(X_train_padded, y_train)

        # Make predictions
        y_pred = stacking_classifier.predict(X_test_padded)

        # Evaluate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        tn, fp, fn, tp = confusion.ravel()
        specificity = tn / (tn + fp)
        FPR = fp / (fp + tn)
        NPV = tn / (tn + fn)

        y_pred_proba = stacking_classifier.predict_proba(X_test_padded)[:, 1]
        auc_roc = roc_auc_score(y_test, y_pred_proba)

        # Print and display the results
        print("Result of Stacking Ensemble")
        print("Accuracy: {:.4f}".format(accuracy))
        print("Precision: {:.4f}".format(precision))
        print("Recall: {:.4f}".format(recall))
        print("F1-Score: {:.4f}".format(f1))
        print("Cohen's Kappa: {:.4f}".format(kappa))
        print(f"Specificity: {specificity:.4f}")
        print(f"MCC: {mcc:.4f}")
        print(f"FPR: {FPR:.4f}")
        print(f"NPV: {NPV:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")

    elif ensembleType == "bagging":
        # Create a bagging ensemble
        bagging_classifier = BaggingClassifier(
            estimator=cnn_model,  # You can use any of the deep learning models here
            n_estimators=10,  # Number of base classifiers
            random_state=42,
            verbose=1
        )

        # Train the bagging ensemble
        bagging_classifier.fit(X_train_padded, y_train)

        # Make predictions
        y_pred = bagging_classifier.predict(X_test_padded)

        # Evaluate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        tn, fp, fn, tp = confusion.ravel()
        specificity = tn / (tn + fp)
        FPR = fp / (fp + tn)
        NPV = tn / (tn + fn)

        y_pred_proba = bagging_classifier.predict_proba(X_test_padded)[:, 1]
        auc_roc = roc_auc_score(y_test, y_pred_proba)

        # Print and display the results
        print("Result of Bagging Ensemble")
        print("Accuracy: {:.4f}".format(accuracy))
        print("Precision: {:.4f}".format(precision))
        print("Recall: {:.4f}".format(recall))
        print("F1-Score: {:.4f}".format(f1))
        print("Cohen's Kappa: {:.4f}".format(kappa))
        print(f"Specificity: {specificity:.4f}")
        print(f"MCC: {mcc:.4f}")
        print(f"FPR: {FPR:.4f}")
        print(f"NPV: {NPV:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")

    elif ensembleType == "boosting":
        # Create an AdaBoost classifier with your deep learning models as base estimators
        boosting_classifier = AdaBoostClassifier(
            estimator=cnn_model,  # You can use any of the deep learning models here
            n_estimators=100,  # Number of base classifiers
            random_state=42,
            algorithm='SAMME.R'
        )

        # Train the AdaBoost classifier
        boosting_classifier.fit(X_train_padded, y_train)

        # Make predictions
        y_pred = boosting_classifier.predict(X_test_padded)


        # Evaluate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        tn, fp, fn, tp = confusion.ravel()
        specificity = tn / (tn + fp)
        FPR = fp / (fp + tn)
        NPV = tn / (tn + fn)

        y_pred_proba = boosting_classifier.predict_proba(X_test_padded)[:, 1]
        auc_roc = roc_auc_score(y_test, y_pred_proba)

        # Print and display the results
        print("Result of Boosting Ensemble")
        print("Accuracy: {:.4f}".format(accuracy))
        print("Precision: {:.4f}".format(precision))
        print("Recall: {:.4f}".format(recall))
        print("F1-Score: {:.4f}".format(f1))
        print("Cohen's Kappa: {:.4f}".format(kappa))
        print(f"Specificity: {specificity:.4f}")
        print(f"MCC: {mcc:.4f}")
        print(f"FPR: {FPR:.4f}")
        print(f"NPV: {NPV:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")

    elif ensembleType == "voting":
        # Create a voting ensemble classifier
        voting_classifier = VotingClassifier(
            estimators=[
                ('ann', ann_model),
                ('cnn', cnn_model),
                ('lstm', lstm_model),
                #('rnn', rnn_model),
                ('gru', gru_model),
                ('bilstm', biLSM_model)
            ],
            voting='soft'  # 'hard' for majority voting, 'soft' for weighted voting
        )

        # Train the voting ensemble classifier
        voting_classifier.fit(X_train_padded, y_train)

        # Make predictions
        y_pred = voting_classifier.predict(X_test_padded)

        # Evaluate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        tn, fp, fn, tp = confusion.ravel()
        specificity = tn / (tn + fp)
        FPR = fp / (fp + tn)
        NPV = tn / (tn + fn)

        y_pred_proba = voting_classifier.predict_proba(X_test_padded)[:, 1]
        auc_roc = roc_auc_score(y_test, y_pred_proba)

        # Print and display the results
        print("Result of Voting Ensemble")
        print("Accuracy: {:.4f}".format(accuracy))
        print("Precision: {:.4f}".format(precision))
        print("Recall: {:.4f}".format(recall))
        print("F1-Score: {:.4f}".format(f1))
        print("Cohen's Kappa: {:.4f}".format(kappa))
        print(f"Specificity: {specificity:.4f}")
        print(f"MCC: {mcc:.4f}")
        print(f"FPR: {FPR:.4f}")
        print(f"NPV: {NPV:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")


def ensembleSpamBase(data, ensembleType):
    X = data.iloc[:, 0:-1].values  # All columns excepts the last one for labels
    y = data.iloc[:, -1].values  # The last column with labels

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Standardize the test for feature scalling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)


    # Define your deep learning models
    def create_ann_model():
        model = Sequential()
        model.add(Dense(units=40, input_dim=57, activation='relu'))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=16, activation='relu'))
        model.add(Dense(units=2, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
        return model

    def create_cnn_model(X_train):
        # Reshape the data for compatibility with the CNN model
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        model = Sequential()
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
        return model

    def create_lstm_model(X_train):
        model = Sequential()
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
        return model

    def create_gru_model(X_train):
        # Initialize the GRU model
        model = Sequential()

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
        return model

    def create_biLSTM_model(X_train):
        # Initialize the LSTM model
        model = Sequential()
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
        return model

    # Create instances of your deep learning models using KerasClassifier wrapper
    ann_model = KerasClassifier(model=create_ann_model, epochs=30, batch_size=16, verbose=0)
    cnn_model = KerasClassifier(model=create_cnn_model(X_train), epochs=8, batch_size=16, verbose=0)
    lstm_model = KerasClassifier(model=create_lstm_model(X_train), epochs=5, batch_size=32, verbose=0)
    #rnn_model = KerasClassifier(model=create_rnn_model, epochs=5, batch_size=16, verbose=0)
    gru_model = KerasClassifier(model=create_gru_model(X_train), epochs=5, batch_size=32, verbose=0)
    biLSM_model = KerasClassifier(model=create_biLSTM_model(X_train), epochs=5, batch_size=32, verbose=0)

    if ensembleType == "stacking":
        # Create a stacking ensemble with MLP (ANN), Random Forest, and SVM as meta-classifier
        base_classifiers = [
            ('ann', ann_model),
            ('cnn', cnn_model),
            ('lstm', lstm_model),
            #('rnn', rnn_model),
            ('gru', gru_model),
            ('bilstm', biLSM_model)
        ]

        stacking_classifier = StackingClassifier(
            estimators=base_classifiers,
            final_estimator=RandomForestClassifier(n_estimators=50),
            stack_method='predict_proba'  # Use 'predict_proba' to get class probabilities for the meta-classifier
        )

        # Train the stacking ensemble
        stacking_classifier.fit(X_train, y_train)

        # Make predictions
        y_pred = stacking_classifier.predict(X_test)

        # Evaluate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        tn, fp, fn, tp = confusion.ravel()
        specificity = tn / (tn + fp)
        FPR = fp / (fp + tn)
        NPV = tn / (tn + fn)

        y_pred_proba = stacking_classifier.predict_proba(X_test)[:, 1]
        auc_roc = roc_auc_score(y_test, y_pred_proba)

        # Print and display the results
        print("Result of Stacking Ensemble")
        print("Accuracy: {:.4f}".format(accuracy))
        print("Precision: {:.4f}".format(precision))
        print("Recall: {:.4f}".format(recall))
        print("F1-Score: {:.4f}".format(f1))
        print("Cohen's Kappa: {:.4f}".format(kappa))
        print(f"Specificity: {specificity:.4f}")
        print(f"MCC: {mcc:.4f}")
        print(f"FPR: {FPR:.4f}")
        print(f"NPV: {NPV:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")

    elif ensembleType == "bagging":
        # Create a bagging ensemble
        bagging_classifier = BaggingClassifier(
            estimator=cnn_model,  # You can use any of the deep learning models here
            n_estimators=10,  # Number of base classifiers
            random_state=42,
            verbose=1
        )

        # Train the bagging ensemble
        bagging_classifier.fit(X_train, y_train)

        # Make predictions
        y_pred = bagging_classifier.predict(X_test)

        # Evaluate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        tn, fp, fn, tp = confusion.ravel()
        specificity = tn / (tn + fp)
        FPR = fp / (fp + tn)
        NPV = tn / (tn + fn)

        y_pred_proba = bagging_classifier.predict_proba(X_test)[:, 1]
        auc_roc = roc_auc_score(y_test, y_pred_proba)

        # Print and display the results
        print("Result of Bagging Ensemble")
        print("Accuracy: {:.4f}".format(accuracy))
        print("Precision: {:.4f}".format(precision))
        print("Recall: {:.4f}".format(recall))
        print("F1-Score: {:.4f}".format(f1))
        print("Cohen's Kappa: {:.4f}".format(kappa))
        print(f"Specificity: {specificity:.4f}")
        print(f"MCC: {mcc:.4f}")
        print(f"FPR: {FPR:.4f}")
        print(f"NPV: {NPV:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")

    elif ensembleType == "boosting":
        # Create an AdaBoost classifier with your deep learning models as base estimators
        boosting_classifier = AdaBoostClassifier(
            estimator=cnn_model,  # You can use any of the deep learning models here
            n_estimators=100,  # Number of base classifiers
            random_state=42,
            algorithm='SAMME.R'
        )

        # Train the AdaBoost classifier
        boosting_classifier.fit(X_train, y_train)

        # Make predictions
        y_pred = boosting_classifier.predict(X_test)


        # Evaluate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        tn, fp, fn, tp = confusion.ravel()
        specificity = tn / (tn + fp)
        FPR = fp / (fp + tn)
        NPV = tn / (tn + fn)

        y_pred_proba = boosting_classifier.predict_proba(X_test)[:, 1]
        auc_roc = roc_auc_score(y_test, y_pred_proba)

        # Print and display the results
        print("Result of Boosting Ensemble")
        print("Accuracy: {:.4f}".format(accuracy))
        print("Precision: {:.4f}".format(precision))
        print("Recall: {:.4f}".format(recall))
        print("F1-Score: {:.4f}".format(f1))
        print("Cohen's Kappa: {:.4f}".format(kappa))
        print(f"Specificity: {specificity:.4f}")
        print(f"MCC: {mcc:.4f}")
        print(f"FPR: {FPR:.4f}")
        print(f"NPV: {NPV:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")

    elif ensembleType == "voting":
        # Create a voting ensemble classifier
        voting_classifier = VotingClassifier(
            estimators=[
                ('ann', ann_model),
                ('cnn', cnn_model),
                ('lstm', lstm_model),
                #('rnn', rnn_model),
                ('gru', gru_model),
                ('bilstm', biLSM_model)
            ],
            voting='soft'  # 'hard' for majority voting, 'soft' for weighted voting
        )

        # Train the voting ensemble classifier
        voting_classifier.fit(X_train, y_train)

        # Make predictions
        y_pred = voting_classifier.predict(X_test)

        # Evaluate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        tn, fp, fn, tp = confusion.ravel()
        specificity = tn / (tn + fp)
        FPR = fp / (fp + tn)
        NPV = tn / (tn + fn)

        y_pred_proba = voting_classifier.predict_proba(X_test)[:, 1]
        auc_roc = roc_auc_score(y_test, y_pred_proba)

        # Print and display the results
        print("Result of Voting Ensemble")
        print("Accuracy: {:.4f}".format(accuracy))
        print("Precision: {:.4f}".format(precision))
        print("Recall: {:.4f}".format(recall))
        print("F1-Score: {:.4f}".format(f1))
        print("Cohen's Kappa: {:.4f}".format(kappa))
        print(f"Specificity: {specificity:.4f}")
        print(f"MCC: {mcc:.4f}")
        print(f"FPR: {FPR:.4f}")
        print(f"NPV: {NPV:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")