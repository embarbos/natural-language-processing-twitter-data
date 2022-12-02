#%%
import json
import pandas as pd
import nltk

# %% NLP PRE-PROCESSING
import re

cleaned_tweets = []
words = []
for tweet in dataremoved['tweet']:
    clean = re.sub(r"(http[s]?\://\S+)|([\[\(].*[\)\]])|([#@]\S+)|\n", " ", tweet)
    clean = re.sub(r"\d", '', clean)
    clean = re.sub(r"'\S+", '', clean)
    clean = clean.replace('.', '').replace(';', '').lower()
    words += re.findall(r"(?:\w+|'|’)+", clean)
    cleaned_tweets.append(clean)
    
    
# removing other symbols
corpus = [[re.sub('[^a-zA-Z ]', ' ', document)] for document in cleaned_tweets]
#tokenizing
corpus_tokenized = [nltk.word_tokenize(document[0]) for document in corpus]
# stop words
stopwords = nltk.corpus.stopwords.words("english")
corpus_tokenized = [[word for word in document if word not in stopwords] for document in corpus_tokenized]
#lemmatizing
nltk.download('wordnet')
corpus_lemmatized = [[nltk.WordNetLemmatizer().lemmatize(word) for word in document] for document in corpus_tokenized]
#stitching back together
corpus = [' '.join(document) for document in corpus_lemmatized]

#%% VECTORIZING CORPUS

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Binarizer

vec = CountVectorizer(tokenizer = nltk.word_tokenize)

freq = vec.fit_transform(corpus)
ohot = Binarizer().fit_transform(freq)
corpus_binary = ohot.todense()

corpus_binary = np.asarray(corpus_binary)

#%% LDA

from sklearn.decomposition import LatentDirichletAllocation

ntopics = 2
lda = LatentDirichletAllocation(n_components = ntopics, learning_method = 'online')

lda.fit(corpus_binary)

posterior = lda.transform(corpus_binary)


#%%

lda.components_

wordTopics = pd.DataFrame(lda.components_.T, index = vec.get_feature_names_out())


wordTopics = wordTopics.apply(lambda x: x / sum(x), 1)
wordTopics = wordTopics.reset_index()
wordTopics.columns = ['word'] + ['topic ' + str(i) for i in range(0,ntopics)]



wordTopics.sort_values(by = 'topic 1', ascending = False)['word'].iloc[1:10]
wordTopics.sort_values(by = 'topic 0', ascending = False)['word'].iloc[1:10]













