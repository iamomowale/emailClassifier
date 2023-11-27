import tensorflow as tf
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)
import re
import email.parser
import email.policy
import os
from bs4 import BeautifulSoup
import nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
#from nltk import WordNetLemmatizer





def getRawEmail(emailDataPath):
    mails = os.listdir(emailDataPath)
    all_email = []
    for file1 in mails:
        with open(f"{emailDataPath}{file1}",'rb') as file:
            all_email.append(email.parser.BytesParser(policy=email.policy.default).parse(file))
    return all_email

#Remove HTML Tags
def removeHTMLtags(mail):
    try:
        soup = BeautifulSoup(mail.get_payload(), 'html.parser')
        return(soup.text.replace('\n\n',''))
    except:
        return None

#Convert to plain text
def Convert_Mails(mail):
    for section in mail.walk():
        content_type = section.get_content_type()
        if (content_type in ['text/plain', 'text/html']):
            try:
                content = section.get_payload()
            except:
                content = str(section.get_payload())
            if (content_type=='text/plain'):
                return (content)
            else:
                return (removeHTMLtags(section))
        else:
            continue

#Convert each email to dataframe
def Create_Dataframe(mails, labelTitle, label):
    converted_email = []
    for mail in range(len(mails)):
        converted_email.append(Convert_Mails(mails[mail]))
    dataframe = pd.DataFrame(converted_email, columns=['emails'])
    dataframe[labelTitle]=label
    return (dataframe)

#Join dataframe
def join_dataframe(df_1, df_2):
    return (pd.concat([df_1,df_2], axis=0))

#remove missing values and escape sequence
def clean_email(email):
    cleaned_email = re.sub(r"[^a-zA-Z0-9]+", ' ', email)
    return cleaned_email

def removeMissingValues(emails):
    emails = emails.dropna()
    emails = emails.sample(frac=1).reset_index(drop=True)
    emails['emails']= emails['emails'].apply(clean_email)
    return emails

#Tokenize
def Tokenized_Mail(email):
    email['emails'] = email['emails'].apply(lambda x: x.lower())
    email['emails'] = email['emails'].apply(word_tokenize)
    return email

#Lemmatize
def Lemmatize_Mail(email):
    lemmatizer = WordNetLemmatizer()
    email['emails'] = [[lemmatizer.lemmatize(word) for word in l] for l in email['emails']]
    email['emails'] = email['emails'].apply(lambda token_list: " ".join(token for token in token_list))
    return email

#Convert .csv to dataframe (Enron)
def csvToDataframe(csvPath):
    data = pd.read_csv(csvPath)
    data['Spam'] = data['Category'].apply(lambda x: 1 if x == 'spam' else 0)
    data.rename(columns={'Message':'emails'}, inplace=True) #Renaming the column name to match others
    data = data.drop(['Category'], axis=1) #Removing the column after changing it 0 and 1
    return data
