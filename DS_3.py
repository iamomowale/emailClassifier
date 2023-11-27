from preProcessing import getRawEmail, Create_Dataframe, join_dataframe, removeMissingValues, Tokenized_Mail, Lemmatize_Mail
from ANN_CNN_RNN_model import annModel, cnnModel, rnnModel
from LSTM_GRU_BiLSTM_model import LSTMmodel, GRUmodel, BiLSTMmodel
from EnsembleModel import ensembleModels

ham_path = "Phising_Corpus/SpamAssassin/all_ham/"
spam_path = "Phising_Corpus/SpamAssassin/spam/spam_2/"
phish_path = "Phising_Corpus/public_phishing/phishing3/"
all_ham_mail = getRawEmail(ham_path)
all_spam_mail = getRawEmail(spam_path)
all_phish_mail = getRawEmail(phish_path)

#Remove HTML tags and convert to Pandas Dataframe
ham_df = Create_Dataframe(all_ham_mail,'Spam/Phish', 0)
spam_df = Create_Dataframe(all_spam_mail,'Spam/Phish', 1)
phish_df = Create_Dataframe(all_phish_mail,'Spam/Phish', 1)

#Joining both Ham and Spam together to one Dataframe
cleaned_data = join_dataframe(ham_df,spam_df)
cleaned_data = join_dataframe(cleaned_data,phish_df)

#Remove Missing Data
cleaned_data = removeMissingValues(cleaned_data)

#Tokenize
cleaned_data = Tokenized_Mail(cleaned_data)

#Lemmatize
cleaned_data = Lemmatize_Mail(cleaned_data)
#print(cleaned_data)

"""Uncomment any classifier or ensemble model to run it"""
#Classifier
#annModel(cleaned_data,"Spam/Phish")
#cnnModel(cleaned_data, "Spam/Phish")
#LSTMmodel(cleaned_data, "Spam/Phish")
#GRUmodel(cleaned_data, "Spam/Phish")
#rnnModel(cleaned_data, "Spam/Phish")
#BiLSTMmodel(cleaned_data, "Spam/Phish")

#Ensemble models
#ensembleModels(cleaned_data, "Spam/Phish", "voting")
#ensembleModels(cleaned_data, "Spam/Phish", "boosting")
#ensembleModels(cleaned_data, "Spam/Phish", "bagging")
#ensembleModels(cleaned_data, "Spam/Phish", "stacking")

