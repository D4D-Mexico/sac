#!/usr/bin/env python
# coding=utf-8
#  Author: Angela Chapman
#  Date: 8/6/2014
#
#  This file contains code to accompany the Kaggle tutorial
#  "Deep learning goes to the movies".  The code in this file
#  is for Part 1 of the tutorial on Natural Language Processing.
#
# *************************************** #

import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.append( os.path.join(os.path.dirname(__file__), 'DeepLearningMovies/' ))
from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
from pandas import DataFrame 
import numpy as np
from bs4 import BeautifulSoup
import re
import MySQLdb
import pandas.io.sql as sql
from nltk.corpus import stopwords

# connect
db_conn = MySQLdb.connect(host="localhost", user="root", passwd="", db="sacdb")
cursor = db_conn.cursor()
cursor.execute('SET NAMES utf8')

query_portal = "SELECT folioSAC,tema_id,CONCAT_WS('_',nomTema,nomsubtema) as nomTema_nomsubtema,descripcion FROM portal,peticion,temas WHERE folioSAC=folioPeticion AND temaId=tema_id"
all_portal = sql.read_sql(query_portal,db_conn).dropna()
query_temas = "SELECT tema_id FROM peticion GROUP BY tema_id ORDER BY tema_id"
temas = sql.read_sql(query_temas,db_conn).dropna()

def review_words( raw_text ):
    review_text = BeautifulSoup(raw_text).get_text()
    letters_only = re.sub("^(\w+)[0-9]@", " ", review_text) 
    callback = lambda pat: pat.group(0).decode('utf-8').lower()
    iac = re.sub(u"Ă",u"í",letters_only)
    ene = re.sub(u"Ñ",u"ñ",iac)
    words = re.sub("(\w+)", callback, ene).split()
    stops = set(stopwords.words("spanish")) 
    meaningful_words = [w for w in words if not w in stops] 
    return( u" ".join( meaningful_words ))  

if __name__ == '__main__':
    train=None
    test=None
    for i in temas["tema_id"].tolist():
        num_lines = 0
        num_lines = len(all_portal[all_portal.tema_id==i])
        train0 = all_portal[all_portal.tema_id==i].head(num_lines-num_lines/2)
        train0.loc[train0['tema_id'] != 82, 'tema_id'] = 0
        train = train0 if train is None else train.append(train0)
        test0 = all_portal[all_portal.tema_id==i].tail(num_lines/2)
        test0.loc[test0['tema_id'] != 82, 'tema_id'] = 0
        test = test0 if test is None else test.append(test0)


    train.reset_index(inplace=True,drop=True)
    test.reset_index(inplace=True,drop=True)

    print train.shape
    print test.shape
    print 'The first descripcion is:'
    print review_words(train["descripcion"][0])
    print review_words(train["descripcion"][1])
    print review_words(test["descripcion"][0])
    print review_words(test["descripcion"][1])

    raw_input("Press Enter to continue...")


    print 'Download text data sets. If you already have NLTK datasets downloaded, just close the Python download window...'
    #nltk.download()  # Download text data sets, including stop words

    # Initialize an empty list to hold the clean descripcions
    clean_train_descripcions = []

    # Loop over each descripcion; create an index i that goes from 0 to the length
    # of the movie descripcion list

    print "Cleaning and parsing the training set movie descripcions...\n"
    for i in xrange( 0, len(train["descripcion"])):
        clean_text_train = review_words( train["descripcion"][i] )
        train["nomTema_nomsubtema"][i] = re.sub(" ","",train["nomTema_nomsubtema"][i] )
        clean_train_descripcions.append(u" ".join(KaggleWord2VecUtility.review_to_wordlist(clean_text_train, True)))


    # ****** Create a bag of words from the training set
    #
    print "Creating the bag of words...\n"


    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
    vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.
    train_data_features = vectorizer.fit_transform(clean_train_descripcions)

    # Numpy arrays are easy to work with, so convert the result to an
    # array
    train_data_features = train_data_features.toarray()

    # ******* Train a random forest using the bag of words
    #
    print "Training the random forest (this may take a while)..."


    # Initialize a Random Forest classifier with 100 trees
    forest = RandomForestClassifier(n_estimators = 200)

    # Fit the forest to the training set, using the bag of words as
    # features and the sentiment labels as the response variable
    #
    # This may take a few minutes to run
    forest = forest.fit( train_data_features, train["tema_id"] )



    # Create an empty list and append the clean descripcions one by one
    clean_test_descripcions = []

    print "Cleaning and parsing the test set movie descripcions...\n"
    for i in xrange(0,len(test["descripcion"])):
        clean_text_test = review_words( test["descripcion"][i] )
        test["nomTema_nomsubtema"][i] = re.sub(" ","",test["nomTema_nomsubtema"][i] )
        clean_test_descripcions.append(u" ".join(KaggleWord2VecUtility.review_to_wordlist(clean_text_test, True)))

    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test_descripcions)
    test_data_features = test_data_features.toarray()

    # Use the random forest to make sentiment label predictions
    print "Predicting test labels...\n"
    result = forest.predict(test_data_features)

    # Copy the results to a pandas dataframe with an "id" column and
    # a "sentiment" column
    output = pd.DataFrame( data={"folioSAC":test["folioSAC"], "tema_id":result, "test_id":test["tema_id"], "nomTema_nomsubtema":test["nomTema_nomsubtema"]} )

    # Use pandas to write the comma-separated output file
    output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'Bag_of_Words_model.csv'), index=False, quoting=2,encoding='utf-8')
    print "Wrote results to Bag_of_Words_model.csv"
