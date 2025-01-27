import numpy as np
import pandas as pd
import re
import string
import pickle

#load the model
with open('static/model/model.pickle','rb') as f:
    model=pickle.load(f)

#Accessing stopwords list to remove stopwords
with open('static/model/corpora/stopwords/english','r') as file:
    sw=file.read().splitlines()

vocab=pd.read_csv('static/model/vocabulary.txt',header=None)
tokens=vocab[0].tolist()

#For stemming
from nltk.stem import PorterStemmer
ps=PorterStemmer()

#Remove puctuations
def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def preprocessing(text):
    data = pd.DataFrame([text], columns=['tweet'])
    #convert to lowercase
    data["tweet"]=data["tweet"].apply(lambda x: " ".join(x.lower() for x in x.split()))
    #Remove links
    data["tweet"]=data["tweet"].apply(lambda x: " ".join(re.sub(r'^https?:\/\/.*[\r\n]*','',x,flags=re.MULTILINE) for x in x.split()))
    #Remove numbers
    data["tweet"]=data["tweet"].str.replace('\\d+', '', regex=True)
    #Remove punctuations
    data["tweet"] = data["tweet"].apply(remove_punctuations)
    #Remove stopwords
    data["tweet"]=data["tweet"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
    #Stemming
    data["tweet"]=data["tweet"].apply(lambda x: " ".join(ps.stem(x) for x in x.split()))
    return data['tweet']

def vectorizer(ds):
    vectorized_lst = []  # List to store the vectorized sentences

    # Iterate through each sentence in the dataset
    for sentence in ds:
        sentence_lst = np.zeros(len(tokens))  # Initialize a zero vector of vocabulary size

        # Iterate through the vocabulary
        for i in range(len(tokens)):
            # If the vocabulary word is in the sentence, set the corresponding index to 1
            if tokens[i] in sentence.split():
                sentence_lst[i] = 1

        vectorized_lst.append(sentence_lst)  # Append the vectorized sentence to the list

    # Convert the list of vectors to a NumPy array of float32 type
    vectorized_lst_new = np.asarray(vectorized_lst, dtype=np.float32)
    return vectorized_lst_new

def get_prediction(vectorized_txt):
    prediction=model.predict(vectorized_txt)
    if prediction ==1:
        return 'Negative'
    else:
        return 'Positive'
