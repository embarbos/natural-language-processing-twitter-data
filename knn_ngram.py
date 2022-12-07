# %%
from sklearn.neighbors import KNeighborsClassifier
from knn_classification import data_labelled, data_test

# %%
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(ngram_range=(2, 4))
train_bow = count_vect.fit_transform(data_labelled['tweets'])
test_bow = count_vect.fit_transform(data_test['tweets'])
#getting feature names, this will act as header for BOW data and  will help to recognize important features
feature_names_bow = count_vect.get_feature_names()

# %%
y_train = data_labelled['label']
knn_bow = KNeighborsClassifier(n_neighbors=10, algorithm='brute')
knn_bow.fit(train_bow, y_train)
bow_pred = knn_bow.predict_proba(test_bow)