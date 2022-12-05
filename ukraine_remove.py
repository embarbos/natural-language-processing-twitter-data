#%% REMOVING TWEETS ABOUT UKRAINE CONFLICT

import pandas as pd
import json

f = open(r'twit_data.json', 'rb')
data = [json.loads(line) for line in f]
tweet_df = pd.DataFrame(data)
tweet_df = tweet_df[['tweet','link']]
tweet_df = tweet_df.drop_duplicates()

#%%
dataremoved = data[~data.iloc[:,0].str.contains('Ukraine')]
dataremoved = dataremoved[~data.iloc[:,0].str.contains('ukraine')]
dataremoved = dataremoved[~data.iloc[:,0].str.contains('Ukrainian')]
dataremoved = dataremoved[~data.iloc[:,0].str.contains('ukrainian')]
dataremoved = dataremoved[~data.iloc[:,0].str.contains('Ukrainians')]
dataremoved = dataremoved[~data.iloc[:,0].str.contains('ukrainians')]
dataremoved = dataremoved[~data.iloc[:,0].str.contains('Zaporizhzhia')]
dataremoved = dataremoved[~data.iloc[:,0].str.contains('war')]
dataremoved = dataremoved[~data.iloc[:,0].str.contains('russia')]
dataremoved = dataremoved[~data.iloc[:,0].str.contains('Russia')]
dataremoved = dataremoved[~data.iloc[:,0].str.contains('russian')]
dataremoved = dataremoved[~data.iloc[:,0].str.contains('Russian')]
dataremoved = dataremoved[~data.iloc[:,0].str.contains('russians')]
dataremoved = dataremoved[~data.iloc[:,0].str.contains('Russians')]
# %%
