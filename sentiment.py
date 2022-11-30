#%%
from nlp import new_text
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('vader_lexicon')

# %%
tokenized_words = word_tokenize(new_text, "english")
# %%
score = SentimentIntensityAnalyzer().polarity_scores(new_text)
# %%
score
# %%
