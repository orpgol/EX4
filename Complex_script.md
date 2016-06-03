---
title: "markdown_python"
author: "Orpaz Goldstein"
date: "JUN 3, 2016"
output: html_document
---

Ex4
==================
Crowdflower Search Results Relevance
------------------

#### Add text

```python
from nltk.corpus import stopwords
from nltk.stem.porter import *
import numpy as np
import pandas as pd
import re
import nltk
from sklearn.cross_validation import KFold
import pickle
from bs4 import BeautifulSoup
from sklearn.cross_validation import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
# Add my directory for nltk data
nltk.data.path.append('/Users/orpaz/Developer/nltk_data')
```


```python
def stem_data(data):
    '''
    Helper function to stem the raw training and test data.
    '''
    stemmer = PorterStemmer()

    for i, row in data.iterrows():

        q = (" ").join([z for z in BeautifulSoup(row["query"]).get_text(" ").split(" ")])
        t = (" ").join([z for z in BeautifulSoup(row["product_title"]).get_text(" ").split(" ")]) 
        d = (" ").join([z for z in BeautifulSoup(row["product_description"]).get_text(" ").split(" ")])

        q=re.sub("[^a-zA-Z0-9]"," ", q)
        t=re.sub("[^a-zA-Z0-9]"," ", t)
        d=re.sub("[^a-zA-Z0-9]"," ", d)

        q= (" ").join([stemmer.stem(z) for z in q.split()])
        t= (" ").join([stemmer.stem(z) for z in t.split()])
        d= (" ").join([stemmer.stem(z) for z in d.split()])
        
        data.set_value(i, "query", str(q))
        data.set_value(i, "product_title", str(t))
        data.set_value(i, "product_description", str(d))

def remove_stop_words(data):
    '''
    Helper function to remove stop words
    from the raw training and test data.
    '''
    stop = stopwords.words('english')

    for i, row in data.iterrows():

        q = row["query"].lower().split(" ")
        t = row["product_title"].lower().split(" ")
        d = row["product_description"].lower().split(" ")

        q = (" ").join([z for z in q if z not in stop])
        t = (" ").join([z for z in t if z not in stop])
        d = (" ").join([z for z in d if z not in stop])

        data.set_value(i, "query", q)
        data.set_value(i, "product_title", t)
        data.set_value(i, "product_description", d)

def get_n_gram_string_similarity(s1, s2, n):
    '''
    Helper function to get the n-gram "similarity" between two strings,
    where n-gram similarity is defined as the percentage of n-grams
    the two strings have in common out of all of the n-grams across the
    two strings.
    '''
    s1 = set(get_n_grams(s1, n))
    s2 = set(get_n_grams(s2, n))
    if len(s1.union(s2)) == 0:
        return 0
    else:
        return float(len(s1.intersection(s2)))/float(len(s1.union(s2)))

def get_n_grams(s, n):
    '''
    Helper function that takes in a string and the degree of n gram n and returns a list of all the
    n grams in the string. String is separated by space.
    '''

    token_pattern = re.compile(r"(?u)\b\w+\b")
    word_list = token_pattern.findall(s)
    n_grams = []


    if n > len(word_list):
        return []
    
    for i, word in enumerate(word_list):
        n_gram = word_list[i:i+n]
        if len(n_gram) == n:
            n_grams.append(tuple(n_gram))
    return n_grams

def calculate_nearby_relevance_tuple(group, row, col_name, ngrams):
    '''
    Takes the group of rows for a particular query ("group") and a row within that 
    group ("row") and returns a dictionary of "similarity"  calculations of row compared to the rest 
    of the rows in group. Returns a tuple of calculations that will be used to create similarity features for row.
    '''

    ngrams = range(1, ngrams + 1)
    #Weighted ratings takes the form
    #{median rating : {ngram : [number of comparisons with that rating/ngram, cumulative sum of similarity for that rating/ngram]}}
    weighted_ratings = {rating: {ngram: [0,0] for ngram in ngrams} for rating in range(1,5)}

    for i, group_row in group.iterrows():
        if group_row['id'] != row['id']:

            for ngram in ngrams:
                similarity = get_n_gram_string_similarity(row[col_name], group_row[col_name], ngram)
                weighted_ratings[group_row['median_relevance']][ngram][1] += similarity
                weighted_ratings[group_row['median_relevance']][ngram][0] += 1

    return weighted_ratings

################################################################
################ FEATURE EXTRACTION FUNCTIONS ##################
################################################################

def extract_features(data):
    '''
    Perform feature extraction for variables that can be extracted
    the same way for both training and test data sets. The input
    "data" is the pandas dataframe for the training or test sets.
    '''
    token_pattern = re.compile(r"(?u)\b\w+\b")
    data["query_tokens_in_title"] = 0.0
    data["query_tokens_in_description"] = 0.0
    data["percent_query_tokens_in_description"] = 0.0
    data["percent_query_tokens_in_title"] = 0.0
    for i, row in data.iterrows():
        query = set(x.lower() for x in token_pattern.findall(row["query"]))
        title = set(x.lower() for x in token_pattern.findall(row["product_title"]))
        description = set(x.lower() for x in token_pattern.findall(row["product_description"]))
        if len(title) > 0:
            data.set_value(i, "query_tokens_in_title", float(len(query.intersection(title)))/float(len(title)))
            data.set_value(i, "percent_query_tokens_in_title", float(len(query.intersection(title)))/float(len(query)))
        if len(description) > 0:
            data.set_value(i, "query_tokens_in_description", float(len(query.intersection(description)))/float(len(description)))
            data.set_value(i, "percent_query_tokens_in_description", float(len(query.intersection(description)))/float(len(query)))
        data.set_value(i, "query_length", len(query))
        data.set_value(i, "description_length", len(description))
        data.set_value(i, "title_length", len(title))

        two_grams_in_query = set(get_n_grams(row["query"], 2))
        two_grams_in_title = set(get_n_grams(row["product_title"], 2))
        two_grams_in_description = set(get_n_grams(row["product_description"], 2))

        data.set_value(i, "two_grams_in_q_and_t", len(two_grams_in_query.intersection(two_grams_in_title)))
        data.set_value(i, "two_grams_in_q_and_d", len(two_grams_in_query.intersection(two_grams_in_description)))

def extract_training_and_test_features(train, test):
    '''
    Perform feature extraction for variables that require both 
    training and test data sets for extraction (i.e. you cannot extract the test features without using data from the training set). 
    E.g. features developed include average and relevance for each 
    query in training, and the 1-gram and 2-gram similarity weighted relevance.
    '''
    train_group = train.groupby('query')
    test["q_mean_of_training_relevance"] = 0.0
    test["q_median_of_training_relevance"] = 0.0
    test["avg_relevance_variance"] = 0
    for i, row in train.iterrows():
        group = train_group.get_group(row["query"])
        
        q_mean = group["median_relevance"].mean()
        train.set_value(i, "q_mean_of_training_relevance", q_mean)
        test.loc[test["query"] == row["query"], "q_mean_of_training_relevance"] = q_mean

        q_median = group["median_relevance"].median()
        train.set_value(i, "q_median_of_training_relevance", q_median)
        test.loc[test["query"] == row["query"], "q_median_of_training_relevance"] = q_median

        avg_relevance_variance = group["relevance_variance"].mean()
        train.set_value(i, "avg_relevance_variance", avg_relevance_variance)
        test.loc[test["query"] == row["query"], "avg_relevance_variance"] = avg_relevance_variance

        weight_dict = calculate_nearby_relevance_tuple(group, row, col_name = 'product_title', ngrams = 2)
        for rating in weight_dict:
            for ngram in weight_dict[rating]:
                variable_name = "average_title_" + str(ngram) + "gram_similarity_" + str(rating)
                if weight_dict[rating][ngram][0] != 0:
                    train.set_value(i, variable_name, float(weight_dict[rating][ngram][1])/float(weight_dict[rating][ngram][0]))
                else:
                    train.set_value(i, variable_name, 0)

        weight_dict = calculate_nearby_relevance_tuple(group, row, col_name = 'product_description', ngrams = 2)
        for rating in weight_dict:
            for ngram in weight_dict[rating]:
                variable_name = "average_description_" + str(ngram) + "gram_similarity_" + str(rating)
                if weight_dict[rating][ngram][0] != 0:
                    train.set_value(i, variable_name, float(weight_dict[rating][ngram][1])/float(weight_dict[rating][ngram][0]))
                else:
                    train.set_value(i, variable_name, 0)


    for i, row in test.iterrows():
        group = train_group.get_group(row["query"])

        weight_dict = calculate_nearby_relevance_tuple(group, row, col_name = 'product_title', ngrams = 2)
        for rating in weight_dict:
            for ngram in weight_dict[rating]:
                variable_name = "average_title_" + str(ngram) + "gram_similarity_" + str(rating)
                if weight_dict[rating][ngram][0] != 0:
                    test.set_value(i, variable_name, float(weight_dict[rating][ngram][1])/float(weight_dict[rating][ngram][0]))
                else:
                    test.set_value(i, variable_name, 0)

        weight_dict = calculate_nearby_relevance_tuple(group, row, col_name = 'product_description', ngrams = 2)
        for rating in weight_dict:
            for ngram in weight_dict[rating]:
                variable_name = "average_description_" + str(ngram) + "gram_similarity_" + str(rating)
                if weight_dict[rating][ngram][0] != 0:
                    test.set_value(i, variable_name, float(weight_dict[rating][ngram][1])/float(weight_dict[rating][ngram][0]))
                else:
                    test.set_value(i, variable_name, 0)


def extract(train, test):

    print("Extracting training features")
    extract_features(train)
    print("Extracting test features")
    extract_features(test)

#     print("Extracting features that must use data in the training set for both test and training data extraction")
#     extract_training_and_test_features(train, test)

    y_train = train.loc[:,"median_relevance"]
    train.drop("median_relevance", 1)

    if 'median_relevance' in test.columns.values:
        y_test = test.loc[:, "median_relevance"]
        test.drop("median_relevance", 1)
    else:
        y_test = []

    return train, y_train, test, y_test

def ouput_final_model(model, train, test, features):

    y = train["median_relevance"]
    train_with_features = train[features]
    test_with_features = test[features]
    model.fit(train_with_features, y)
    predictions = model.predict(test_with_features)
    submission = pd.DataFrame({"id": test["id"], "prediction": predictions})
    return (submission, model.score(train_with_features, y))
```


```python
# Load the training file
train = pd.read_csv("train.csv").fillna("")
test  = pd.read_csv("test.csv").fillna("")
```


```python
#stem first
stem_data(train)
stem_data(test)

#remove stop words
remove_stop_words(train)
remove_stop_words(test)

#Extract variables for full train and test set
extract(train, test)
pickle.dump(train, open('train_extracted_df.pkl', 'wb'))
pickle.dump(test, open('test_extracted_df.pkl', 'wb'))  
```

    /usr/local/Cellar/python3/3.5.1/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages/bs4/__init__.py:166: UserWarning: No parser was explicitly specified, so I'm using the best available HTML parser for this system ("lxml"). This usually isn't a problem, but if you run this code on another system, or in a different virtual environment, it may use a different parser and behave differently.
    
    To get rid of this warning, change this:
    
     BeautifulSoup([your markup])
    
    to this:
    
     BeautifulSoup([your markup], "lxml")
    
      markup_type=markup_type))


    Extracting training features
    Extracting test features



```python
#Load all of the data extracted from the separate extraction.py script, including the full train/test data 
#and the StratifiedKFold data
train = pickle.load(open('train_extracted_df.pkl', 'rb'))
test = pickle.load(open('test_extracted_df.pkl', 'rb'))
y_train = train["median_relevance"]

#Name the features to be used in the first 3 models (Random Forest, SVC, Adaboost).
features = ['query_tokens_in_title', 'query_tokens_in_description', 'percent_query_tokens_in_description', 'percent_query_tokens_in_title', 'query_length', 'description_length', 'title_length', 'two_grams_in_q_and_t', 'two_grams_in_q_and_d']

#Random forest model
print("Begin random forest model")
model = RandomForestClassifier(n_estimators=300, n_jobs=1, min_samples_split=10, random_state=1, class_weight='auto')
rf_final_predictions, rf_score = ouput_final_model(model, train, test, features)
pickle.dump(rf_final_predictions, open('rf_final_predictions.pkl', 'wb'))

#SVC
print("Begin SVC model")
scl = StandardScaler()
svm_model = SVC(C=10.0, random_state = 1, class_weight = {1:2, 2:1.5, 3:1, 4:1})
model = Pipeline([('scl', scl), ('svm', svm_model)])
svc_final_predictions, svc_score = ouput_final_model(model, train, test, features)
pickle.dump(svc_final_predictions, open('svc_final_predictions.pkl', 'wb'))

#AdaBoost
print("Begin AdaBoost model")
model = AdaBoostClassifier(n_estimators=200, random_state = 1, learning_rate = 0.25)
adaboost_final_predictions, ada_score = ouput_final_model(model, train, test, features)
pickle.dump(adaboost_final_predictions, open('adaboost_final_predictions.pkl', 'wb'))
```

    Begin random forest model
    Begin SVC model
    Begin AdaBoost model



```python
preds = [rf_final_predictions, svc_final_predictions, adaboost_final_predictions]
#Decided on weights based on the model scores
weights = [0.5,0.3,0.2]
predictions = sum([weights[x] * preds[x]["prediction"].astype(int) for x in range(3)])
predictions = [int(round(p)) for p in predictions]
```


```python
submission = pd.DataFrame({"id": test["id"], "prediction": predictions})
submission.to_csv('ensembled_submission.csv', index=False)
```


```python

```
