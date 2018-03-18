from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import string
from nltk.corpus import stopwords
import math
from collections import Counter
nltk.download('stopwords')
nltk.download('punkt')

stemmer = PorterStemmer()

# Tokenize, stem a document
def tokenize(text):
    
    # remove special characters !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    text = "".join([ch for ch in text if ch not in string.punctuation])
    
    # break up a sentence into a list of words and punctuation with no whitespaces
    # http://www.nltk.org/book/ch03.html
    tokens = nltk.word_tokenize(text)
    
    # convert to lowercase words then have stemmers remove morphological affixes
    # why join them into a string if we are going to split them at the end ?
    # http://www.nltk.org/howto/stem.html
    return " ".join([stemmer.stem(word.lower()) for word in tokens]) 

# compute IDF, storing idf values in a dictionary
def idf_values(vocabulary, documents):
    idf = {}
    num_documents = len(documents)    
    for i, term in enumerate(vocabulary):                
        # compute the number of documents each term appears in the collection 
        idf[term] = math.log(num_documents/sum(term in x for x in documents), math.e)         
    return idf

# Function to generate the vector for a document (with normalisation)
# compute tf-idf
def vectorize(document, vocabulary, idf):
    vector = [0]*len(vocabulary)
    counts = Counter(document)
    max_count = counts.most_common(1)[0][1]
    for i, term in enumerate(vocabulary):
        vector[i] = document.count(term)/max_count * idf[term]
    return vector

# Function to compute cosine similarity
def cosine_similarity(v1,v2):
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i];
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    if sumxy == 0:
            result = 0
    else:
            result = sumxy / (math.sqrt(sumxx)*math.sqrt(sumyy))
    return result