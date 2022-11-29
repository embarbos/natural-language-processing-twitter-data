import json
import pandas as pd

#%%

f = open(r'C:/Users/L S/Documents/Project/twit_data.json', 'rb')

data = [json.loads(line) for line in f]

#%% 

df = pd.DataFrame(data)

df = df[['tweet','link']]

df = df.drop_duplicates()


#%% RANDOMLY PULLING TRAINING TWEETS

traindf = df.sample(frac=0.1, random_state=0)

#%% REMAINDER DATA FRAME

remainderdf = df.loc[~df.index.isin(traindf.index)]

#%% PORTIONING TRAINING SETS

trainingset_1 = traindf[:222]
trainingset_2 = traindf[222:444]
trainingset_3 = traindf[444:666]
trainingset_4 = traindf[666:]

#%% EXPORT TO CSV

trainingset_1.to_csv('trainingset_1.csv',encoding='utf-8-sig', header=None)

trainingset_2.to_csv('trainingset_2.csv',encoding='utf-8-sig', header=None)

trainingset_3.to_csv('trainingset_3.csv',encoding='utf-8-sig', header=None)

trainingset_4.to_csv('trainingset_4.csv',encoding='utf-8-sig', header=None)


remainderdf.to_csv('remainder.csv',encoding='utf-8-sig', header=None)
