import csv
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pre_processing import pStemmer
   

def qaResponse(query):
    THRESHOLD = 0.6
    #Data Access
    df = pd.read_csv('data/COMP3074-CW1-Dataset-QA.csv')
    df['processed_Q'] = df['Question'].apply(pStemmer)
    
    tf = TfidfVectorizer(use_idf=True, sublinear_tf=True, stop_words=stopwords.words('english'))
    
    stemmed_query = pStemmer(query)
    #stemmed_doc =  pStemmer(df['Question'])

    
    tdm = tf.fit_transform(df['Question'])
    bow = tf.transform([query])
    
    #Calculate Similarity between query and all questions (using their vector form)
    cosineSimilarities = cosine_similarity(tdm, bow).flatten()
    related_docs_indices = cosineSimilarities.argsort()[:-2:-1]
    #If similarity > threshold, return answer from dataset
    if (cosineSimilarities[related_docs_indices] > THRESHOLD):
        resp = [df['Answer'][i] for i in related_docs_indices[:1]]
        return resp
    else:
        ERROR_MESSAGE = 'I am sorry, I cannot help you with this one. Hope to in the future. Cheers :)'
        return ERROR_MESSAGE

