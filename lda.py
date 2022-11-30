#%%
from nlp import standardized, cleaned_tweets
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Binarizer
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
nltk.download('punkt')

#%%
vec = CountVectorizer(tokenizer = nltk.word_tokenize)
freq = vec.fit_transform(cleaned_tweets)
ohot = Binarizer().fit_transform(freq)
corpus_binary = ohot.todense()
corpus_binary = np.asarray(corpus_binary)

#%%
ntopics = 2
lda = LatentDirichletAllocation(n_components = ntopics, learning_method = 'online')
lda.fit(corpus_binary)
posterior = lda.transform(corpus_binary)

#%%
df = pd.DataFrame(posterior).reset_index()
df.columns = ['tweet'] + ['topic ' + str(i) for i in range(0,ntopics)]
# %%
df
# %%