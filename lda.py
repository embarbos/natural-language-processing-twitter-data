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
freq = vec.fit_transform(standardized)
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
wordTopics = pd.DataFrame(lda.components_.T, index = vec.get_feature_names_out())
wordTopics = wordTopics.apply(lambda x: x / sum(x), 1)
wordTopics = wordTopics.reset_index()
wordTopics.columns = ['word'] + ['topic ' + str(i) for i in range(0,ntopics)]
wordTopics
# %%
wordTopics.sort_values(by = 'topic 1', ascending = False)['word'].iloc[1:50]
# %%
wordTopics.sort_values(by = 'topic 0', ascending = False)['word'].iloc[1:50]
# %%
