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

##### The task was to find how relevant a search result is to a query.
##### My first approach was to replace similar words in the query and results to the same words. Using Word2Vec, i changed all high similarity words to be the same words. but that wasn't very helpfull. 
##### Next i used the similarity score between the whole query and the results, which gave me better results.
##### So i wanted a better way to find similarities, but also to improve the previous results.

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

##### This function i used to Stem the data before running any feature extraction to remove unneccesery words suffixes, and there for improve results.
```python
def stem_data(data):
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
```
##### Next, i used a stop words removal function, to remove all unwanted words that we should not consider when feature extracting.
```python
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
```
##### The next step was to improve the similarity between query and product title, and query and description. This function returnes the n gram of a string. That is all the possible combinations of n tokens of the string (which i can later compre)
```python
def get_n_gram_string_similarity(s1, s2, n):
    s1 = set(get_n_grams(s1, n))
    s2 = set(get_n_grams(s2, n))
    if len(s1.union(s2)) == 0:
        return 0
    else:
        return float(len(s1.intersection(s2)))/float(len(s1.union(s2)))

def get_n_grams(s, n):
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
```

##### The next two function are my feature extraction. They prepare the dataset and create the collumns to be later filled, and also perform small similarity checkes on the dataset, like tokenizing the query description and title, and finding intersections between them.

```python
def extract_features(data):
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

def extract(train, test):

    print("Extracting training features")
    extract_features(train)
    print("Extracting test features")
    extract_features(test)

    y_train = train.loc[:,"median_relevance"]
    train.drop("median_relevance", 1)

    if 'median_relevance' in test.columns.values:
        y_test = test.loc[:, "median_relevance"]
        test.drop("median_relevance", 1)
    else:
        y_test = []

    return train, y_train, test, y_test
```

##### function to return the final prediction output of the models i will use later.
```python
def ouput_final_model(model, train, test, features):

    y = train["median_relevance"]
    train_with_features = train[features]
    test_with_features = test[features]
    model.fit(train_with_features, y)
    predictions = model.predict(test_with_features)
    submission = pd.DataFrame({"id": test["id"], "prediction": predictions})
    return (submission, model.score(train_with_features, y))
```

##### Loading the data sets
```python
# Load the training file
train = pd.read_csv("train.csv").fillna("")
test  = pd.read_csv("test.csv").fillna("")
```

##### First thing is to stem the dataset and remove stop words before any extraction is made. Then i can extract the features of both train and test.
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



##### The data was saved in each step to prevent double calculations. So here we load the previous step results. Then i define 3 different models to create an ensamble, declare the features i will be using, and then send each model to be fitted and to predict the test set. The results are saved for each model. 
##### Later i decided to add a weight to each model, so i started returning the scores of each model and then will decide on an appropreate weight for each model.
```python
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


##### Here we take the predictions from each model, assign a weight based on the previous step score results for each model, and then apply the weight while ensambling the results from the 3 models i used to recieve a final result.
```python
preds = [rf_final_predictions, svc_final_predictions, adaboost_final_predictions]
#Decided on weights based on the model scores
weights = [0.5,0.3,0.2]
predictions = sum([weights[x] * preds[x]["prediction"].astype(int) for x in range(3)])
predictions = [int(round(p)) for p in predictions]
```

##### sumbission file preparations 

```python
submission = pd.DataFrame({"id": test["id"], "prediction": predictions})
submission.to_csv('orpaz_submission.csv', index=False)
```


```python

```
