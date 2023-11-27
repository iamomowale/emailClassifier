from preProcessing import getRawEmail, Create_Dataframe, join_dataframe, removeMissingValues, Tokenized_Mail, Lemmatize_Mail
from ANN_CNN_RNN_model import annModel, cnnModel, rnnModel
from LSTM_GRU_BiLSTM_model import LSTMmodel, GRUmodel, BiLSTMmodel
from EnsembleModel import ensembleModels

ham_path = "Phising_Corpus/SpamAssassin/all_ham/"
spam_path = "Phising_Corpus/SpamAssassin/spam/spam_2/"
all_ham_mail = getRawEmail(ham_path)
all_spam_mail = getRawEmail(spam_path)

#Remove HTML tags and convert to Pandas Dataframe
ham_df = Create_Dataframe(all_ham_mail,'Spam', 0)
spam_df = Create_Dataframe(all_spam_mail, 'Spam', 1)

#Joining both Ham and Spam together to one Dataframe
cleaned_data = join_dataframe(ham_df,spam_df)
#print(cleaned_data.head(5))

#Remove Missing Data
cleaned_data = removeMissingValues(cleaned_data)
#print(cleaned_data.head(5))

#Tokenize
cleaned_data = Tokenized_Mail(cleaned_data)
#print(cleaned_data.head(5))

#Lemmatize
cleaned_data = Lemmatize_Mail(cleaned_data)
#print(cleaned_data.head(5))

"""Uncomment any classifier or ensemble model to run it"""
#Classifier
#annModel(cleaned_data,"Spam")
#cnnModel(cleaned_data, "Spam")
#LSTMmodel(cleaned_data, "Spam")
#GRUmodel(cleaned_data, "Spam")
#rnnModel(cleaned_data, "Spam")
#BiLSTMmodel(cleaned_data, "Spam")

#Ensemble models
#ensembleModels(cleaned_data, "Spam", "voting")
#ensembleModels(cleaned_data, "Spam", "boosting")
#ensembleModels(cleaned_data, "Spam", "bagging")
#ensembleModels(cleaned_data, "Spam", "stacking")