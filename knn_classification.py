#%%
import pandas as pd
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

#%%
# read the file
colnames=['tweets', 'label']
data = pd.read_csv("final_141B/trainingset_1_lemmatized.csv", names=colnames, header=None)

#%%
# one-hot encoding
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Binarizer
import nltk

# 1. create a CountVectorizer
vec = CountVectorizer(tokenizer = nltk.word_tokenize)
# 2. fit_transform
# convert the type of "tweets" to str
data["tweets"] = data["tweets"].astype(str)
tweet_list = list(data['tweets'])
freq = vec.fit_transform(tweet_list)
# create one-hot encoding
ohot = Binarizer().fit_transform(freq)
# one-hot encoding
corpus_binary = ohot.todense()

# convert matrix to dataframe
encoder_df = pd.DataFrame(corpus_binary)

# create x and y for knn
x = encoder_df
y = data['label']


#%%
# test the optimal k for the accuracy
k_range = range(1, 31)
k_error = []
k_acc = []
optimal_k = 0
min_error = 1
max_acc = 0

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # the value of cv decides the ratio of training and testing data
    scores = cross_val_score(knn, x, y, cv=4, scoring='accuracy')
    # error rate
    error_rate = 1 - scores.mean()
    # record the best performance with value of k
    if error_rate < min_error:
        min_error = error_rate
        optimal_k = k
        max_acc = scores.mean()
    k_error.append(error_rate)
    # accuracy rate
    k_acc.append(scores.mean())

# plot: x is the k value, y is the error value
plt.plot(k_range, k_error)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Error')
plt.show()
print("the optimal k is: ", optimal_k)

#%%
# check the effect of random_state for accuracy

# Import train_test_split function
from sklearn.model_selection import train_test_split
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Split dataset into training set and test set
# test which test_size produces the largest acc
test_size_range = range(0.1,0.4,0.1)
acc = 0
test_size_acc = []
test_size = []
for size in test_size_range:
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=size, random_state=0) # 80% training and 20% test
    # Create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=25)

    # Train the model using the training sets
    knn.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = knn.predict(X_test)
    # Model Accuracy, how often is the classifier correct?
    # calculate the accuracy
    acc_s = metrics.accuracy_score(y_test, y_pred)
    if acc <= acc_s:
        acc = acc_s
        print("test_size: ", size)
        print("Accuracy:", acc)
        test_size_acc.append(acc)
        test_size.append(size)

plt.plot(test_size, test_size_acc)
plt.xlabel('Test Size')
plt.ylabel('acc for KNN')
plt.show()
