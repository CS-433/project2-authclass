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


#################################################################

def character_ngrams(feed, collection, kind):
    feed_lower = feed.lower()
    lhs = ['feed_lower.count(\''] * len(collection)
    rhs = ['\')'] * len(collection)
    char_ngrams = []
    for p1, p2, p3 in zip(lhs, collection, rhs):
        command = p1 + p2 + p3
        if (p2 != ('\\' + string.punctuation[6])) & (p2 != ('\\\\')):
            command = command.replace('\\', '', 1)
        char_ngrams.append(eval(command))
    num_chars = sum(char_ngrams)
    if num_chars == 0:
        num_chars = 1
    return [x / num_chars for x in char_ngrams]

def character_ngrams_wrapper(dataframe, textcolumn, newcolumn, n, kind='letter'):
    print("Performing " + kind + " " + str(n) + "-gram...")
    baseline = time.time()
    if kind == 'letter':
        collection = list(string.ascii_lowercase)
    if kind == 'digit':
        collection = list(string.digits)
    if kind == 'punctuation':
        collection_temp = list(string.punctuation)
        collection = ['\\' + a for a in collection_temp]
    if n == 2:
        collection = [a + b for a in collection for b in collection]
    if n == 3:
        collection = [a + b + c for a in collection for b in collection for c in collection]
    if n == 4:
        collection = [a + b + c + d for a in collection for b in collection for c in collection for d in collection]
    dataframe[newcolumn] = dataframe[textcolumn].apply(character_ngrams, args = (collection,kind))
    print("Performed " + kind + " " + str(n) + "-gram in " + str(time.time() - baseline) + " seconds")

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
              
def character_count_proportion_wrapper(dataframe, textcolumn, kind='letter'):
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
    dataframe[kind + '_count_prop'] = dataframe[textcolumn].apply(character_count_proportion, args = (collection,kind))
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

def word_count_wrapper(dataframe, textcolumn):
    print("Performing word count...")
    baseline = time.time()
    dataframe['word_count'] = dataframe[textcolumn].apply(word_count)
    print("Performed word count in " + str(time.time() - baseline) + " seconds")

#################################################################

def word_length_avg(feed):
    words = word_list(feed, "letters_only")
    word_lengths = []
    for word in words:
        word_lengths.append(len(word))
    return sum(word_lengths) / len(word_lengths)
    
def word_length_avg_wrapper(dataframe, textcolumn):
    print("Performing word length avg...")
    baseline = time.time()
    dataframe['word_length_avg'] = dataframe[textcolumn].apply(word_length_avg)
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

def word_length_distribution_wrapper(dataframe, textcolumn):
    print("Performing word length distribution...")
    baseline = time.time()
    dataframe['word_length_distribution'] = dataframe[textcolumn].apply(word_length_distribution)
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

def letter_case_distribution_wrapper(dataframe, textcolumn):
    print("Performing letter case distribution...")
    baseline = time.time()
    dataframe['letter_case_distribution'] = dataframe[textcolumn].apply(letter_case_distribution)
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

def word_case_distribution_wrapper(dataframe, textcolumn):
    print("Performing word case distribution...")
    baseline = time.time()
    dataframe['word_case_distribution'] = dataframe[textcolumn].apply(word_case_distribution)
    print("Performed word case distribution in " + str(time.time() - baseline) + " seconds")
    
#################################################################

def misspelled_prop(feed, spell_instance):
    words = word_list(feed, "inner_punc_allowed")
    misspellings = 0
    for word in words:
            if word != spell_instance(word):
                misspellings += 1
    return misspellings / len(words)
        
def misspelled_prop_wrapper(dataframe, textcolumn, newcolumn):
    print("Performing misspellings proportion...")
    baseline = time.time()
    spell = Speller(lang='en')
    dataframe[newcolumn] = dataframe[textcolumn].progress_apply(misspelled_prop, args = (spell,))
    print("Performed misspellings proportion in " + str(time.time() - baseline) + " seconds")
    
#################################################################

def word_ngrams(feed, collection):
    tw = len(collection)
    feed_lower = feed.lower()
    words = word_list(feed, "letters_only")
    kept_words = []
    for word in words:
        if word in collection:
            kept_words.append(word)
    kept_words = kept_words + collection
    kept_words.sort()
    words_ngram_distro_temp = list(Counter(kept_words).values())
    words_ngram_distro = [(x-1)/tw for x in words_ngram_distro_temp]
    return words_ngram_distro

def word_ngrams_wrapper(dataframe, textcolumn, newcolumn, n):
    print("Performing word " + str(n) + "-gram...")
    baseline = time.time()
    sp = spacy.load('en_core_web_sm')
    stop_words = list(sp.Defaults.stop_words)
    if n == 1:
        all_words = []
        for feed in dataframe[textcolumn]:
            all_words.append(word_list(feed, "letters_only"))
        all_words_lower = []
        for wordlist in all_words:
            for word in wordlist:
                all_words_lower.append(word.lower())
        c = Counter(all_words_lower)
        keep_counter = 0
        loop_counter = 0
        collection = []
        while keep_counter < 200:
            curr_word = c.most_common()[loop_counter][0]
            if curr_word not in stop_words:
                keep_counter += 1
                collection.append(curr_word)
            loop_counter += 1
    dataframe[newcolumn] = dataframe[textcolumn].progress_apply(word_ngrams, args = (collection,))
    print("Performed word " + str(n) + "-gram in " + str(time.time() - baseline) + " seconds")
    
#################################################################
#################################################################
#################################################################
#################################################################
#################################################################

#def lemmatizer(feed, lemmatizer_instance):
#    tokenized features have lemma_ as already. return this if needed.

#def lemmatizer_wrapper(dataframe, textcolumn):
#    lemmatizer_instance = WordNetLemmatizer()
#    if textcolumn + '_lemmatized' not in dataframe.columns:
#        print("Lemmatizing feeds...")
#        baseline = time.time()
#        dataframe[textcolumn + '_lemmatized'] = dataframe[textcolumn].apply(lemmatizer, args=(lemmatizer_instance,))
#        print("Feeds lemmatized in " + str(time.time() - baseline) + " seconds")
#    else:
#        print("Feeds already lemmatized")
    
#################################################################

def tokenizer(feed, spacy_load):
    return spacy_load(feed)

def tokenizer_wrapper(dataframe, textcolumn, spacy_load = None):
    if spacy_load == None:
        spacy_load = spacy.load('en_core_web_sm')
    if textcolumn + '_tokenized' not in dataframe.columns:
        print("Tokenizing feeds...")
        baseline = time.time()
        dataframe[textcolumn + '_tokenized'] = dataframe[textcolumn].apply(tokenizer, args=(spacy_load,))
        print("Feeds tokenized in " + str(time.time() - baseline) + " seconds")
    else:
        print("Feeds already tokenized")

#################################################################

def POS_tags_ngram(tokens, n, collection):
    tc = len(tokens)
    tags = []
    for token in tokens:
        #print(f'{token.text:{12}} {token.pos_:{10}} {token.tag_:{8}} {spacy.explain(token.tag_)}')
        tags.append(token.tag_)
    ngrams = []
    for i in range(tc-(n-1)):
        ngrams.append(tags[i:i+n])
    ngrams_sortable = []
    for i in range(len(ngrams)):
        ngrams_sortable.append('||'.join(ngrams[i]))
    ngrams_sortable = ngrams_sortable + collection
    ngrams_sortable.sort()
    tags_ngram_distro_temp = list(Counter(ngrams_sortable).values())
    tags_ngram_distro = [(x-1)/tc for x in tags_ngram_distro_temp]
    return tags_ngram_distro

def POS_tags_ngram_wrapper(dataframe, tokencolumn, newcolumn, n):
    sp = spacy.load('en_core_web_sm')
    print("Performing POS tags " + str(n) + "-grams...")
    baseline = time.time()
    collection = sp.pipe_labels['tagger']
    if n == 1:
        pass
    if n == 2:
        collection = [a + "||" + b for a in collection for b in collection]
    if n == 3:
        collection = [a + "||" + b + "||" + c for a in collection for b in collection for c in collection]
    dataframe[newcolumn] = dataframe[tokencolumn].apply(POS_tags_ngram, args=(n, collection))
    print("Performed POS tags " + str(n) + "-gram in " + str(time.time() - baseline) + " seconds")
    
#################################################################

def stop_words_proportion(tokens, collection):
    tc = len(tokens)
    stop_words = 0
    for token in tokens:
        if token.text.lower in collection:
            stop_words += 1
    return stop_words / tc

def stop_words_proportion_wrapper(dataframe, tokencolumn, newcolumn):
    sp = spacy.load('en_core_web_sm')
    print("Performing stop words ratio...")
    baseline = time.time()
    collection = list(sp.Defaults.stop_words)
    dataframe[newcolumn] = dataframe[tokencolumn].apply(stop_words_proportion, args = (collection,))
    print("Performed stop words ratio in " + str(time.time() - baseline) + " seconds")
    
#################################################################

def hapax_legomena_proportion(tokens, kind, collection):
    token_texts = []
    for token in tokens:
        if str(token.text.lower) not in collection:
            token_texts.append(token.text)
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

def hapax_legomena_proportion_wrapper(dataframe, tokencolumn, newcolumn, kind):
    print("Performing hapax legomena proportion of " + kind + " tokens...")
    sp = spacy.load('en_core_web_sm')
    collection = list(sp.Defaults.stop_words)
    baseline = time.time()
    dataframe[newcolumn] = dataframe[tokencolumn].apply(hapax_legomena_proportion, args = (kind,collection))
    print("Performed hapax legomena proportion of " + kind + " tokens in " + str(time.time() - baseline) + " seconds")

#################################################################

def token_type_ratio(tokens, collection):
    token_texts = []
    for token in tokens:
        if str(token.text.lower) not in collection:
            token_texts.append(token.text)
    total = len(token_texts)
    unique = len(list(set(token_texts)))
    if total == 0:
        return 0
    else:
        return unique / total

def token_type_ratio_wrapper(dataframe, tokencolumn, newcolumn):
    print("Performing token type ratio...")
    baseline = time.time()
    sp = spacy.load('en_core_web_sm')
    collection = list(sp.Defaults.stop_words)
    dataframe[newcolumn] = dataframe[tokencolumn].apply(token_type_ratio, args = (collection,))
    print("Performed token type ratio in " + str(time.time() - baseline) + " seconds")
    
#################################################################