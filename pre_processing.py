import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

## Once Installed Can COMMENT OUT ==> IF NOT ALREADY INSTALLED, UNCOMMENT
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('universal_tagset')
# nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def textTokenize(text):
    text_tokens = word_tokenize(text)
    tokens_without_sw = [token.lower() for token in text_tokens if token not in stopwords.words('english')] 
    return tokens_without_sw



def sbStemmer(text):
    tokens = textTokenize(text)
    stemmer = SnowballStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    stemmed_text = " ".join(stemmed_tokens)
    return stemmed_text


def pStemmer(text):
    tokens = textTokenize(text)
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    stemmed_text = " ".join(stemmed_tokens)
    return stemmed_text

def lemmatizer(text):
    
    lemmatiser = WordNetLemmatizer()
    posmap = {
        'ADJ': 'j',
        'ADV': 'r',
        'NOUN': 'n',
        'VERB': 'v'
    }

    post = nltk.pos_tag(text, tagset='universal')
    #print(post)
    lem_text = []
    for token in post:
        word = token[0]
        tag = token[1]
        if tag in posmap.keys():
            lem_text.append([lemmatiser.lemmatize(word), posmap[tag]])
        else:
            lem_text.append(lemmatiser.lemmatize(word))
            
    return lem_text


