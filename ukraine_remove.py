#%% REMOVING TWEETS ABOUT UKRAINE CONFLICT

import pandas as pd

data = pd.read_csv(r'*.csv',header=None)

#%%
dataremoved = data[~data.iloc[:,1].str.contains('Ukraine')]
dataremoved = dataremoved[~data.iloc[:,1].str.contains('Ukrainian')]
dataremoved = dataremoved[~data.iloc[:,1].str.contains('Ukrainians')]
dataremoved = dataremoved[~data.iloc[:,1].str.contains('Zaporizhzhia')]

dataremoved.to_csv('*.csv',encoding='utf-8-sig', header=None)
