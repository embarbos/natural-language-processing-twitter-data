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


# %% define one-hot encoding for data
def one_hot_encoding_tweets(data):
    # 1. create a CountVectorizer
    vec = CountVectorizer(tokenizer = nltk.word_tokenize)
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

    return x,y


# create x, y for knn
x = one_hot_encoding_tweets(data)[0]
y = one_hot_encoding_tweets(data)[1]

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


# %%
# predict lables for test data using training data
def pred_label_for_test(train_data, data_test):
    merge_data = train_data.append(data_test, ignore_index=True)

    # %% for
    x = one_hot_encoding_tweets(merge_data)[0]
    y = one_hot_encoding_tweets(merge_data)[1]
    
    # convert 'negative', 'neutral', 'positive' into 3,2,1     
    enc = LabelEncoder()
    label_encoder = enc.fit(y)
    y = label_encoder.transform(y)

    # %% for labelled.csv and remainder.csv
    X_train = x.loc[:1174, :]
    y_train = y[:1175]

    X_test = x.loc[1175:, :]

    # Create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=10)

    # Train the model using the training sets
    knn.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = knn.predict(X_test)
    
    return y_pred, y_train


# drawing bar plots for data
def bar_plot_for_data(data_for_bar):
    req = np.array(np.unique(data_for_bar, return_counts=True)).T
    freq_data = sorted(
                        [(name, float(val)) for name, val in freq],
                         key=lambda x:x[1],
                         reverse=True
                        )

    colors_list = ['Red', 'Orange', 'Blue']
    p1 = plt.bar(*zip(*freq_data), color=colors_list)

    n = len(data_for_bar)
    for rect1 in p1:
        height = rect1.get_height()
        plt.annotate("{}%".format(round(height/n, 2)), (rect1.get_x() + rect1.get_width()/2,
                                            height+.05), ha="center", va="bottom", fontsize=15)

    plt.show()


# %% fix k=10, predict labels of test data with cleaned and uncleand data
colnames = ['tweets', 'label']
data_labelled = pd.read_csv('labelled.csv', names=colnames, header=None)
data_labelled_cleaned = pd.read_csv('labelled_cleaned.csv', names=colnames, header=None)
colnames = ['tweets']
data_test = pd.read_csv("remainder.csv", names=colnames, header=None)
data_test['label'] = ""   # add one empty column "label"

# obtain the predication labels
y_pred_labelled = pred_label_for_test(data_labelled, data_test)[0]
y_pred_labelled_cleand = pred_label_for_test(data_labelled_cleaned, data_test)[0]

# draw barplots
bar_plot_for_data(y_pred_labelled)
bar_plot_for_data(y_pred_labelled_cleand)


# %% predict lables for test data using clean and uncleaned lda vectors
# bar plots with percentages
def pred_label_for_lda(train_lda, test_lda)
    # %% for labelled_LDA_vectors.csv and remainder_LDA_vectors.csv
    X_train = train_lda.loc[:1174, :1]
    y_train = train_lda.loc[:1174, 2]
    enc = LabelEncoder()
    label_encoder = enc.fit(y_train)
    y_train = label_encoder.transform(y_train)+1

    X_test = test_lda.loc[1175:, :]

    # Create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=10)

    # Train the model using the training sets
    knn.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = knn.predict(X_test)


# %% predict labels of test data with cleaned and uncleand lda data
train_data_stopw = pd.read_csv("labelled_LDA_vectors_withStopwords.csv", header=None, skiprows=1)
test_data_stopw = pd.read_csv("remainder_LDA_vectors_withStopwords.csv", header=None, skiprows=1)

train_data = pd.read_csv("labelled_LDA_vectors.csv", header=None, skiprows=1)
test_data = pd.read_csv("remainder_LDA_vectors.csv", header=None, skiprows=1)

# obtain the predication labels
y_pred_lda = pred_label_for_lda(train_data_stopw, test_data_stopw)[0]
y_pred_lda_cleand = pred_label_for_lda(train_data, test_data)[0]

# draw barplots
bar_plot_for_data(y_pred_lda)
bar_plot_for_data(y_pred_lda_cleand)




