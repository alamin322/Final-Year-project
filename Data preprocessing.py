import numpy as np
import pandas as pd
import re
import string

import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pickle
import json
import os

ps = PorterStemmer()

cv = CountVectorizer(max_features=1200)


def simplePreprocessing(text):
    patt = re.compile(pattern=r"[\t:\d\.]")
    a = patt.sub(repl="", string=text)
    b = a.translate(str.maketrans('', '', string.punctuation))
    c = " ".join(b.split())
    return c


def removeSingleAlphabet(text):
    patt = re.compile(pattern=r" (mm|mm x|w mm|mp|x|w|mp) ")
    patt2 = re.compile(pattern=r" (ww|x x|f|w|mm|n m|a l v|n) ")
    patt3 = re.compile(pattern=r" (q|v|l|hz – khz|uv|hr|pu) ")
    mypatt = [patt, patt2, patt3]

    for i in mypatt:
        text = i.sub(repl=" ", string=text)
    return text


def textPorterStemmer(text):
    words = word_tokenize(text=text)
    words = [ps.stem(word=word) for word in words if word not in stopwords.words('english')]
    text = ' '.join(words)
    return text


def full_preprocessing():
    df = pd.read_csv("datasets/newdf.csv")
    df[['name', 'description', 'brand', 'Model']] = df[['name', 'description', 'brand', 'Model']].applymap(
        func=lambda x: x.lower())
    df['description'] = df['description'].apply(func=simplePreprocessing)
    df['description'] = df['description'].apply(func=removeSingleAlphabet)
    df['tags'] = df['name'] + df['description'] + df['brand'] + df['Model']
    df = df.drop(columns=['description', 'brand', 'Model'])
    cv_vectors = cv.fit_transform(raw_documents=df['tags']).toarray()
    cv_similarity = cosine_similarity(X=cv_vectors)
    if os.path.exists("Model And New dataset/model.pkl") and os.path.exists("Model And New dataset/updatedDf.pkl"):
        os.remove("Model And New dataset/model.pkl")
        os.remove("Model And New dataset/updatedDf.pkl")
        pickle.dump(obj=cv_similarity, file=open(file="Model And New dataset/CV_similarity.pkl", mode='wb'))
        pickle.dump(obj=df.to_dict(), file=open(file="Model And New dataset/updatedDf.pkl", mode='wb'))
    else:
        pickle.dump(obj=cv_similarity, file=open(file="Model And New dataset/CV_similarity.pkl", mode='wb'))
        pickle.dump(obj=df.to_dict(), file=open(file="Model And New dataset/updatedDf.pkl", mode='wb'))


# Function call
full_preprocessing()
