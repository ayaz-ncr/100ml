# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 22:27:50 2019

@author: ayaz

the purpose of this code was to analyze free flow text using different methods

"""

# bishop book article analysis

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# Importing the dataset
#dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
#dataset = pd.read_csv('fie.txt', sep=' ',header=None, error_bad_lines=False)


# the following line works best with newline delimited text file
lines1 = [line for line in open('bishop-prml-book.txt').read().splitlines() if line]
print(len(lines1))

'''
text1 = open('bishop-prml-book.txt').read()
text2 = open('bishop-prml-book.txt').readline()
text3 = open('bishop-prml-book.txt').readlines()

with open('bishop-prml-book.txt', 'r') as file:
    text4 = file.read().replace('\n', '')
 
with open('bishop-prml-book.txt', 'r') as infile:
    for line in infile:
        text5 = infile.read().replace('\n', '')    
'''

'''
# this reads whole file without splitting on lines, but this will not be useful later
# as we do not get a term-document matrix with this   
with open('bishop-prml-book.txt', 'rb') as file:
    text1 = file.read().replace('\n', '')    

corpus = []

import re
text2 = re.sub('[^a-zA-Z]', ' ', text1)
text2 = text2.lower()
text2 = text2.split()

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
text3 = [w for w in text2 if not w in stop_words] 

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
text4 = [ps.stem(word) for word in text3]
text5 = ' '.join(text4)
corpus.append(text5)
'''

'''
from nltk.tokenize import sent_tokenize, word_tokenize 
sentences1 = sent_tokenize(text2) # whole string in one sentence, not what was required
words1 = word_tokenize(text2)
'''


  
# Cleaning the texts
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, len(lines1)):
    book = re.sub('[^a-zA-Z]', ' ', lines1[i])
    book = book.lower()
    book = book.split()
    ps = PorterStemmer()
    book = [ps.stem(word) for word in book if not word in set(stopwords.words('english'))]
    book = ' '.join(book)
    corpus.append(book)

print(len(corpus))




# Creating the Bag of Words model
# 1. using CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X1 = cv.fit_transform(corpus).toarray()
#y = lines1.iloc[:, 1].values

print(cv.get_feature_names())
print(len(cv.get_feature_names()))


true_k = 10
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X1)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = cv.get_feature_names()
#terms = vectorizer.get_feature_names()
for i in range(true_k):
    print "Cluster %d:" % i,
    for ind in order_centroids[i, :10]:
        print ' %s' % terms[ind],
    print


import gensim 
from gensim import corpora

# dictionary needs tokens not strings
corpus1 = [d.split() for d in corpus]

dictionary = corpora.Dictionary(corpus1)
print("\n dictionary => ",dictionary)

doc_term_matrix = [dictionary.doc2bow(doc) for doc in corpus1]
print("\n doc_term_matrix => ",doc_term_matrix)

Lda = gensim.models.ldamodel.LdaModel

ldamodel = Lda(doc_term_matrix, num_topics = 50, id2word = dictionary, passes = 50)

topics = ldamodel.print_topics()

print("\n topics => ",topics)


# 2. using tfidf instead of countvectorizer

from sklearn.feature_extraction.text import TfidfVectorizer    
vectorizer = TfidfVectorizer(stop_words='english')
X2 = vectorizer.fit_transform(corpus).toarray()

true_k = 10
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X2)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
#terms = vectorizer.get_feature_names()
for i in range(true_k):
    print "Cluster %d:" % i,
    for ind in order_centroids[i, :10]:
        print ' %s' % terms[ind],
    print


# display wordcloud
    
#from wordcloud import WordCloud, STOPWORDS
#import matplotlib.pyplot as plt
#stopwords = set(STOPWORDS)
from wordcloud import WordCloud
def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
    background_color='white',
    #stopwords=stopwords,
    max_words=200,
    max_font_size=40, 
    scale=3,
    random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()

show_wordcloud(corpus)