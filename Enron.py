from preProcessing import csvToDataframe, removeMissingValues, Tokenized_Mail, Lemmatize_Mail
from ANN_CNN_RNN_model import annModel, cnnModel, rnnModel
from LSTM_GRU_BiLSTM_model import LSTMmodel, GRUmodel, BiLSTMmodel
from EnsembleModel import ensembleModels

#Convert the CSV to Datframe and label ham as 0 and spam as 1

cleaned_data = csvToDataframe("Enron/mail_data.csv")

#Remove Missing Data
#cleaned_data = removeMissingValues(cleaned_data)
#print(cleaned_data.head(5))

#Tokenize
cleaned_data = Tokenized_Mail(cleaned_data)
#print(cleaned_data.head(5))

#Lemmatize
cleaned_data = Lemmatize_Mail(cleaned_data)
#print(cleaned_data.head(5))

"""Uncomment any classifier or ensenble model to run it"""
#Classifier
#annModel(cleaned_data,"Spam")
#cnnModel(cleaned_data, "Spam")
#LSTMmodel(cleaned_data, "Spam")
#GRUmodel(cleaned_data, "Spam")
#BiLSTMmodel(cleaned_data, "Spam")
#rnnModel(cleaned_data,"Spam")

#Ensemble models
#ensembleModels(cleaned_data, "Spam", "voting")
#ensembleModels(cleaned_data, "Spam", "boosting")
#ensembleModels(cleaned_data, "Spam", "bagging")
#ensembleModels(cleaned_data, "Spam", "stacking")

