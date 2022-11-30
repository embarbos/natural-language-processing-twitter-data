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
