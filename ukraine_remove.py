#%% REMOVING TWEETS ABOUT UKRAINE CONFLICT

import pandas as pd
import json

f = open(r'twit_data.json', 'rb')
data = [json.loads(line) for line in f]
tweet_df = pd.DataFrame(data)
tweet_df = tweet_df[['tweet','link']]
tweet_df = tweet_df.drop_duplicates()

#%%
dataremoved = tweet_df[~tweet_df.iloc[:,1].str.contains('Ukraine')]
dataremoved = dataremoved[~tweet_df.iloc[:,1].str.contains('Ukrainian')]
dataremoved = dataremoved[~tweet_df.iloc[:,1].str.contains('Ukrainians')]
dataremoved = dataremoved[~tweet_df.iloc[:,1].str.contains('Zaporizhzhia')]
# %%
