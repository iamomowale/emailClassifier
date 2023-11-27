import tensorflow as tf
import numpy as np
import pandas as pd
from ANN_CNN_RNN_model import ANNspamBase, CNNspamBase, rnnSpambase
from LSTM_GRU_BiLSTM_model import LSTMspamBase, GRUspamBase, BiLSTMspamBase
from EnsembleModel import ensembleSpamBase

#This is a benchmark dataset
#Which means it is already cleaned with feature extracted

extracted_data = pd.read_csv('spambase/spambase.data', sep=',')
#print(extracted_data.tail(10))
#print(type(extracted_data))
#print(extracted_data.shape)

"""Uncomment any classifier or ensemble model to run it"""
#Classifiers
#ANNspamBase(extracted_data)
#CNNspamBase(extracted_data)
#LSTMspamBase(extracted_data)
#GRUspamBase(extracted_data)
#rnnSpambase(extracted_data)
#BiLSTMspamBase(extracted_data)

#Ensemble Models
#ensembleSpamBase(extracted_data, "voting")
#ensembleSpamBase(extracted_data, "boosting")
#ensembleSpamBase(extracted_data, "bagging")
#ensembleSpamBase(extracted_data, "stacking")


