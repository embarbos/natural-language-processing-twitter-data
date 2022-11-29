#%%
import nltk
import json
import pandas as pd
import re

#%%
f = open(r'twit_data.json', 'rb')
data = [json.loads(line) for line in f]
tweet_df = pd.DataFrame(data)
tweet_df = tweet_df[['tweet','link']]
tweet_df = tweet_df.drop_duplicates()

# %%
cleaned_tweets = []
for tweet in tweet_df['tweet'][:10]:
    cleaned_tweets += re.findall(r"(?:\w+|')+", tweet)
# %%
