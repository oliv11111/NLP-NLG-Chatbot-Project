import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



def stResponse(query):
    THRESHOLD = 0.5


    df = pd.read_csv('data/COMP3074-CW1-Dataset-ST.csv')
    questions = df['Question']
    responses = df['Answer']


    tf = TfidfVectorizer(use_idf=True, sublinear_tf=True, stop_words=stopwords.words('english'))

    tdm  = tf.fit_transform(questions)
    bow = tf.transform([query])



    

    cosineSimilarities = cosine_similarity(tdm, bow).flatten()
    related_docs_indicies = cosineSimilarities.argsort()[:-2:-1]

    if (cosineSimilarities[related_docs_indicies] > THRESHOLD):
        resp = [responses[i] for i in related_docs_indicies[:1]]
        return resp
    else:
        ERROR_MESSAGE = 'I am sorry, I cannot help you with this one. Hope to in the future. Cheers :)'
        return ERROR_MESSAGE



