import os, json
import pandas as pd
from pathlib import Path
import re
from collections import Counter
import numpy as np
import nltk
import time
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from writeprints.text_processor import Processor
import string
import spacy
from langdetect import DetectorFactory
from langdetect import detect_langs
from sklearn.preprocessing import LabelEncoder
from sklearn import svm, datasets
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import ast
import warnings
from textblob import Word
from tqdm import tqdm
from tqdm.notebook import tqdm
tqdm.pandas()
from autocorrect import Speller
import statistics
from nltk.util import ngrams
import random


#################################################################

def character_ngrams(feed, collection, n):
    ngrams = []
    for token in feed:
        for i in range(len(token)-n+1):
            ngrams.append(token[i:(i+n)].lower())
    ngrams = ngrams + collection
    c = Counter(ngrams)
    result = []
    for ngram in collection:
        result.append(c[ngram])
    len_col = len(collection)
    tot_result = sum(result)
    denom = tot_result - len_col
    if denom == 0:
        denom = 1
    return [(x-1) / denom for x in result]

def character_ngrams_wrapper(dataframe, feed_token_space, newcolumn, n, collection_size, kind='letter', train_collection = None):
    baseline = time.time()
    if train_collection == None:
        print("Performing train " + kind + " " + str(n) + "-gram...")
        # Create list of all tokens across all train feeds
        master_feed = dataframe[feed_token_space].apply(pd.Series).stack().tolist()
        # Create list of viable n-grams based on function arguments
        if kind == 'letter':
            collection = list(string.ascii_lowercase)
        if kind == 'digit':
            collection = list(string.digits)
        if kind == 'punctuation':
            collection_temp = list(string.punctuation)
            collection_temp.append('“')
            collection_temp.append('”')
            collection_temp.append('’')
            collection = collection_temp
        if n == 2:
            collection = [a + b for a in collection for b in collection]
        if n == 3:
            collection = [a + b + c for a in collection for b in collection for c in collection]
        if n == 4:
            collection = [a + b + c + d for a in collection for b in collection for c in collection for d in collection]
        # Calculate all possible n-grams from master_feed
        ngrams = []
        for token in master_feed:
            for i in range(len(token)-n+1):
                ngram = token[i:(i+n)].lower()
                if ngram in collection:
                    ngrams.append(ngram.lower())
        # Calculate frequencies, keep only the top collection_size
        c = Counter(ngrams).most_common()
        upper = collection_size
        if len(c) < upper:
            upper = len(c)
        upto50_collection = []
        for i in range(upper):
            upto50_collection.append(c[i][0])
        upto50_collection.sort()  
    if train_collection != None:
        print("Performing test " + kind + " " + str(n) + "-gram...")
        upto50_collection = train_collection
    print(upto50_collection)
    dataframe[newcolumn] = dataframe[feed_token_space].progress_apply(character_ngrams, args = (upto50_collection,n))
    # Return upto50_collection for later use with test set - unless this function WAS being called on the test set!
    if train_collection == None:   
        print("Performed train " + kind + " " + str(n) + "-gram in " + str(time.time() - baseline) + " seconds")
        print("Returned up to 50 most common " + kind + " " + str(n) + "-grams for feature extraction on test set.")
        return upto50_collection
    else:
        print("Performed test " + kind + " " + str(n) + "-gram in " + str(time.time() - baseline) + " seconds")

#################################################################

def multchar_ngrams(list_of_lists_of_strings, collection, n):
    c = Counter("")
    for list_of_strings in list_of_lists_of_strings:
        c = c + Counter(list(ngrams(list_of_strings, n)))
    c = c + Counter(collection)
    result = []
    for ngram in collection:
        result.append(c[ngram])
    len_col = len(collection)
    tot_result = sum(result)
    denom = tot_result - len_col
    if denom == 0:
        denom = 1
    return [(x-1) / denom for x in result]

def word_ngrams_wrapper(dataframe, feed_comment_list_nopunc_lower, newcolumn, n, train_collection = None):
    baseline = time.time()
    if train_collection == None:
        print("Performing train word " + str(n) + "-gram...")
        master_comment_list = dataframe[feed_comment_list_nopunc_lower].apply(pd.Series).stack().tolist()
        c = Counter("")
        for comment in master_comment_list:
            c = c + Counter(list(ngrams(comment, n)))
        c = c.most_common()
        upper = collection_size
        if len(c) < upper:
            upper = len(c)
        upto50_collection = []
        for i in range(upper):
            upto50_collection.append(c[i][0])
        upto50_collection.sort() 
    if train_collection != None:
        print("Performing test word " + str(n) + "-gram...")
        upto50_collection = train_collection
    print(upto50_collection)
    dataframe[newcolumn] = dataframe[feed_comment_list_nopunc_lower].progress_apply(multchar_ngrams, args = (upto50_collection,n))
    # Return upto50_collection for later use with test set - unless this function WAS being called on the test set!
    if train_collection == None:   
        print("Performed train word " + str(n) + "-gram in " + str(time.time() - baseline) + " seconds")
        print("Returned up to 50 most common word" + str(n) + "-grams for feature extraction on test set.")
        return upto50_collection
    else:
        print("Performed test word " + str(n) + "-gram in " + str(time.time() - baseline) + " seconds")
        
#################################################################   
        
def character_count_proportion(feed, collection, kind):
    feed_lower = feed.lower()
    #feed_lower_no_ws = feed.translate({ord(c): None for c in string.whitespace}).lower() # removes spaces and linebreaks
    if kind != "character":
        match = []
        for char in feed_lower:
            if char in collection:
                match.append(1)
            else:
                match.append(0)
        return sum(match), sum(match) / len(match)
    if kind == "character":
        return len(feed_lower), 1
              
def character_count_proportion_wrapper(dataframe, feed_string, kind='letter'):
    print("Performing " + kind + " count & proportion...")
    baseline = time.time()
    if kind == 'letter':
        collection = list(string.ascii_lowercase)
    if kind == 'digit':
        collection = list(string.digits)
    if kind == 'punctuation':
        collection = list(string.punctuation)
    if kind == 'whitespace':
        collection = list(string.whitespace)
    if kind == 'character':
        collection = ['only included for the sake of function call']
    dataframe[kind + '_count_prop'] = dataframe[feed_string].apply(character_count_proportion, args = (collection,kind))
    dataframe[[kind + '_count', kind + '_prop']] = pd.DataFrame(dataframe[kind + '_count_prop'].tolist(), index=dataframe.index)
    dataframe = dataframe.drop([kind + '_count_prop'], axis = 1)
    if kind == "character":
            dataframe = dataframe.drop([kind + '_prop'], axis = 1)
    print("Performed " + kind + " count & proportion in " + str(time.time() - baseline) + " seconds")
    return dataframe

#################################################################

def word_list(feed, kind):
    if kind == "letters_only":
        return re.sub(r'[^a-zA-Z ]+', '', feed).split()
    elif kind == "inner_punc_allowed":
        words = feed.split()
        words_clean = []
        for word in words:
            words_clean.append(word.strip(string.punctuation))
        return words_clean

#################################################################

def word_count(feed):
    words = word_list(feed, "letters_only")
    total_words = len(words)
    return total_words

def word_count_wrapper(dataframe, feed_string):
    print("Performing word count...")
    baseline = time.time()
    dataframe['word_count'] = dataframe[feed_string].apply(word_count)
    print("Performed word count in " + str(time.time() - baseline) + " seconds")

#################################################################

def word_length_avg(feed):
    words = word_list(feed, "letters_only")
    word_lengths = []
    for word in words:
        word_lengths.append(len(word))
    return sum(word_lengths) / len(word_lengths)
    
def word_length_avg_wrapper(dataframe, feed_string):
    print("Performing word length avg...")
    baseline = time.time()
    dataframe['word_length_avg'] = dataframe[feed_string].apply(word_length_avg)
    print("Performed word length avg in " + str(time.time() - baseline) + " seconds")
    
#################################################################

def word_length_distribution(feed):
    words = word_list(feed, "letters_only")
    word_lengths = []
    for word in words:
        word_lengths.append(len(word))
    word_lengths = [x for x in word_lengths if x <= 20]
    wc = len(word_lengths)
    if wc == 0:
        wc = 1
    word_lengths.extend(range(1,21))
    word_lengths.sort()
    length_distro_temp = list(Counter(word_lengths).values())
    length_distro = [(x-1)/wc for x in length_distro_temp]
    return length_distro

def word_length_distribution_wrapper(dataframe, feed_string):
    print("Performing word length distribution...")
    baseline = time.time()
    dataframe['word_length_distribution'] = dataframe[feed_string].apply(word_length_distribution)
    print("Performed word length distribution in " + str(time.time() - baseline) + " seconds")
    
#################################################################

def word_short_prop(word_length_distribution):
    return sum(word_length_distribution[0:3])

def word_short_prop_wrapper(dataframe, word_length_distribution, word_count):
    if word_length_distribution not in dataframe.columns or word_count not in dataframe.columns:
        print("missing required columns")
        return
    print("Performing word count short...")
    baseline = time.time()
    dataframe['word_short_prop'] = dataframe[word_length_distribution].apply(word_short_prop)
    dataframe['word_short_count'] =  dataframe['word_short_prop']*dataframe[word_count]
    print("Performed word count short in " + str(time.time() - baseline) + " seconds")
    
#################################################################

def letter_case_distribution(feed):
    lc_count=0
    for char in feed:
        if(char.islower()):
            lc_count += 1

    uc_count=0
    for char in feed:
        if(char.isupper()):
            uc_count += 1
    
    total = lc_count + uc_count
    return [lc_count/total, uc_count/total]

def letter_case_distribution_wrapper(dataframe, feed_string):
    print("Performing letter case distribution...")
    baseline = time.time()
    dataframe['letter_case_distribution'] = dataframe[feed_string].apply(letter_case_distribution)
    print("Performed letter case distribution in " + str(time.time() - baseline) + " seconds")
    
#################################################################

def word_case_distribution(feed):
    all_lower_word = 0
    all_upper_word = 0
    first_upper_rest_lower_word = 0
    other_word = 0

    # Strips string of everything except letters and spaces
    ## Q: Should we remove single-letter words?? "I" possibly inflating uppercase word proportions...
    words = word_list(feed, "letters_only")
    
    total_words = len(words)
    for word in words:
        char_distro = letter_case_distribution(word)
        if char_distro[0] == 1:
            all_lower_word += 1
        elif char_distro[1] == 1:
            all_upper_word += 1
        else:
            if word[0].isupper() and char_distro[1] == 1/len(word):
                first_upper_rest_lower_word += 1
            else:
                other_word += 1

    return [all_lower_word / total_words, first_upper_rest_lower_word / total_words, all_upper_word / total_words, other_word / total_words]

def word_case_distribution_wrapper(dataframe, feed_string):
    print("Performing word case distribution...")
    baseline = time.time()
    dataframe['word_case_distribution'] = dataframe[feed_string].apply(word_case_distribution)
    print("Performed word case distribution in " + str(time.time() - baseline) + " seconds")
    
#################################################################

def misspelled_prop(feed, spell_instance):
    words = word_list(feed, "inner_punc_allowed")
    misspellings = 0
    for word in words:
            if word != spell_instance(word):
                misspellings += 1
    return misspellings / len(words)
        
def misspelled_prop_wrapper(dataframe, feed_string, newcolumn):
    print("Performing misspellings proportion...")
    baseline = time.time()
    spell = Speller(lang='en')
    dataframe[newcolumn] = dataframe[feed_string].progress_apply(misspelled_prop, args = (spell,))
    print("Performed misspellings proportion in " + str(time.time() - baseline) + " seconds")

#################################################################
#################################################################
#################################################################
#################################################################
#################################################################

#def lemmatizer(feed, lemmatizer_instance):
#    tokenized features have lemma_ as already. return this if needed.

#def lemmatizer_wrapper(dataframe, feed_string):
#    lemmatizer_instance = WordNetLemmatizer()
#    if feed_string + '_lemmatized' not in dataframe.columns:
#        print("Lemmatizing feeds...")
#        baseline = time.time()
#        dataframe[feed_string + '_lemmatized'] = dataframe[feed_string].apply(lemmatizer, args=(lemmatizer_instance,))
#        print("Feeds lemmatized in " + str(time.time() - baseline) + " seconds")
#    else:
#        print("Feeds already lemmatized")
    
#################################################################

def tokenizer(feed_comment_list, spacy_load):
    spacy_list = []
    for comment_tokens in feed_comment_list:
        spacy_list.append(spacy_load(' '.join(comment_tokens)))
    return spacy_list

def tokenizer_wrapper(dataframe, feed_comment_list, spacy_load = None):
    if spacy_load == None:
        spacy_load = spacy.load('en_core_web_sm')
    if feed_comment_list + '_spacy' not in dataframe.columns:
        print("Tokenizing feeds...")
        baseline = time.time()
        dataframe[feed_comment_list + '_spacy'] = dataframe[feed_comment_list].progress_apply(tokenizer, args=(spacy_load,))
        print("Feeds tokenized in " + str(time.time() - baseline) + " seconds")
    else:
        print("Feeds already tokenized")

#################################################################

def feed_comment_list_spacy_tags(feed_comment_list_spacy):
    comments_as_tags = []
    for spacy_doc in feed_comment_list_spacy:
        comment_as_tags = []
        for token in spacy_doc:
            comment_as_tags.append(token.tag_)
        comments_as_tags.append(comment_as_tags)
    return comments_as_tags

def POS_tags_ngram_wrapper(dataframe, feed_comment_list_spacy, newcolumn, n, train_collection = None):
    baseline = time.time()
    if train_collection == None:
        print("Performing train POS tags " + str(n) + "-grams...")
        dataframe[feed_comment_list_spacy + "_tags"] = dataframe[feed_comment_list_spacy].apply(feed_comment_list_spacy_tags)
        master_tag_list = dataframe[feed_comment_list_spacy + "_tags"].apply(pd.Series).stack().tolist()
        c = Counter("")
        for comment in master_tag_list:
            c = c + Counter(list(ngrams(comment, n)))
        c = c.most_common()
        upper = collection_size
        if len(c) < upper:
            upper = len(c)
        upto50_collection = []
        for i in range(upper):
            upto50_collection.append(c[i][0])
        upto50_collection.sort() 
    if train_collection != None:
        print("Performing test POS tags " + str(n) + "-gram...")
        dataframe[feed_comment_list_spacy + "_tags"] = dataframe[feed_comment_list_spacy].apply(feed_comment_list_spacy_tags)
        upto50_collection = train_collection
    print(upto50_collection)
    dataframe[newcolumn] = dataframe[feed_comment_list_spacy + "_tags"].progress_apply(multchar_ngrams, args = (upto50_collection,n))
    # Return upto50_collection for later use with test set - unless this function WAS being called on the test set!
    dataframe = dataframe.drop(feed_comment_list_spacy + "_tags", axis=1)
    if train_collection == None:   
        print("Performed train POS tags " + str(n) + "-gram in " + str(time.time() - baseline) + " seconds")
        print("Returned up to 50 most common POS tags " + str(n) + "-grams for feature extraction on test set.")
        return upto50_collection
    else:
        print("Performed test POS tags " + str(n) + "-gram in " + str(time.time() - baseline) + " seconds")
    
#################################################################

def stop_words_proportion(feed_comment_list_spacy, collection):
    tt = 0
    for comment in feed_comment_list_spacy:
        tt += len(comment)
    stop_words = 0
    for comment in feed_comment_list_spacy:
        for token in comment:
            if token.text.lower() in collection:
                stop_words += 1
    return stop_words / tt

def stop_words_proportion_wrapper(dataframe, feed_comment_list_spacy, newcolumn):
    sp = spacy.load('en_core_web_sm')
    print("Performing stop words ratio...")
    baseline = time.time()
    collection = list(sp.Defaults.stop_words)
    dataframe[newcolumn] = dataframe[feed_comment_list_spacy].apply(stop_words_proportion, args = (collection,))
    print("Performed stop words ratio in " + str(time.time() - baseline) + " seconds")
    
#################################################################

def hapax_legomena_proportion(feed_comment_list_spacy, kind, collection):
    token_texts = []
    for comment in feed_comment_list_spacy:
        for token in comment:
            lower_token_text = token.text.lower()
            if lower_token_text not in collection:
                token_texts.append(lower_token_text)
    if kind == 'total':
        denominator = len(token_texts)
    elif kind == 'unique':
        denominator = len(list(set(token_texts)))
    c = Counter(token_texts)
    hapleg_token = 0
    for token in c:
        if c[token] == 1:
            hapleg_token += 1
    if denominator == 0:
        return 0
    else:
        return hapleg_token / denominator

def hapax_legomena_proportion_wrapper(dataframe, feed_comment_list_spacy, newcolumn, kind):
    print("Performing hapax legomena proportion of " + kind + " tokens...")
    sp = spacy.load('en_core_web_sm')
    collection = list(sp.Defaults.stop_words)
    baseline = time.time()
    dataframe[newcolumn] = dataframe[feed_comment_list_spacy].apply(hapax_legomena_proportion, args = (kind,collection))
    print("Performed hapax legomena proportion of " + kind + " tokens in " + str(time.time() - baseline) + " seconds")

#################################################################

def token_type_ratio(feed_comment_list_spacy, collection):
    token_texts = []
    for comment in feed_comment_list_spacy:
        for token in comment:
            lower_token_text = token.text.lower()
            if lower_token_text not in collection:
                token_texts.append(lower_token_text)
    total = len(token_texts)
    unique = len(list(set(token_texts)))
    if total == 0:
        return 0
    else:
        return unique / total

def token_type_ratio_wrapper(dataframe, feed_comment_list_spacy, newcolumn):
    print("Performing token type ratio...")
    baseline = time.time()
    sp = spacy.load('en_core_web_sm')
    collection = list(sp.Defaults.stop_words)
    dataframe[newcolumn] = dataframe[feed_comment_list_spacy].apply(token_type_ratio, args = (collection,))
    print("Performed token type ratio in " + str(time.time() - baseline) + " seconds")
    
#################################################################