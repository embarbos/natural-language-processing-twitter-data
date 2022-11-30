#%%
import nltk
import json
import pandas as pd
import re
nltk.download('stopwords')

#%%
f = open(r'twit_data.json', 'rb')
data = [json.loads(line) for line in f]
tweet_df = pd.DataFrame(data)
tweet_df = tweet_df[['tweet','link']]
tweet_df = tweet_df.drop_duplicates()

# %%
cleaned_tweets = []
words = []
for tweet in tweet_df['tweet']:
    clean = re.sub(r"(http[s]?\://\S+)|([\[\(].*[\)\]])|([#@]\S+)|\n", "", tweet)
    clean = re.sub(r"\d", '', clean)
    clean = re.sub(r"'\S+", '', clean)
    clean = clean.replace('.', '').replace(';', '').lower()
    words += re.findall(r"(?:\w+|'|’)+", clean)
    cleaned_tweets.append(clean)
    

#%%
stopwords = nltk.corpus.stopwords.words("english")
# %%
standardized = [w for w in words if w not in stopwords]
# %%

#%%
# Remove prefixes and suffixes from words through lemmatization
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# all pos_tags for lower words
twit_tags = nltk.pos_tag(standardized)

# classify "n, a, v, r, "
from nltk.corpus import wordnet


def wordnet_pos(tag):
    """Map a Brown POS tag to a WordNet POS tag."""

    table = {"N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV, "J": wordnet.ADJ}

    # Default to a noun.
    # if the key doesn't exist, it will return a wordnet.NOUN
    return table.get(tag[0], wordnet.NOUN)


# obtain tags
tags = [wordnet_pos(x[1]) for x in twit_tags]


from textblob import TextBlob

# create a new sentence
new_text = " ".join(w for w in standardized)
blob = TextBlob(new_text)
# obtain tags
tags = [wordnet_pos(x[1]) for x in blob.pos_tags]

# finalize the lemmatization
new_text = " ".join(x.lemmatize(t) for x, t in zip(blob.words, tags))
# words after lemmatization
standardized_words = TextBlob(new_text)
#%%

#%%
# Plot the frequency distribution for tokens
fq = nltk.FreqDist(w for w in standardized_words if w.isalnum())
# plot
fq.plot(50)
#%%
