import pandas as pd
import re
from collections import Counter
import time
import string
import spacy
from tqdm import tqdm
#from tqdm.notebook import tqdm
tqdm.pandas()
from autocorrect import Speller
from nltk.util import ngrams
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import statistics
import random as random
from random import sample
from sklearn.preprocessing import LabelEncoder

################################################################# EXTRACT FROM JUPYTER

# Raw string version of feeds
def feed_string(feed):
    print(feed)
    return ' '.join(feed)

## Doing this via loop because two columns involved in function instead of one...
def create_comment_word_indices(comment_lengths):
    np_comment_lengths = np.array(comment_lengths)
    return np.cumsum(np_comment_lengths)

# List-of-comments w/ punctuation stripped and lowercase applied version of feeds
def strip_punc_and_lower_nested_list(feed_comment_list):
    feed_comment_list_nopunc_lower = []
    for comment in feed_comment_list:
        feed_comment_list_nopunc_lower.append(re.sub(r'[^A-Za-z0-9 ]+', '', ' '.join(comment)).lower().split()) 
    return feed_comment_list_nopunc_lower

def build_cohorts(n_native,n_nonnat,eng_native,eng_nonnat,seed):
    ''' BUild three cohorts : all, native and non-native '''
    eng_native_sample = eng_native.sample(n_native, random_state=seed)
    eng_nonnat_sample = eng_nonnat.sample(n_nonnat, random_state=seed)    

    # Before splitting into cohorts, perform all pre-feature-extraction processing
    eng_feeds = pd.concat([eng_nonnat_sample, eng_native_sample], ignore_index=False, axis=0) # 
    eng_feeds['author'] = eng_feeds.index
    eng_feeds = pd.wide_to_long(eng_feeds, ["feed", "slices"], i="author", j="intra_author_feed_id").sort_index()
    eng_feeds = eng_feeds.rename(columns={"slices": "comment_lengths", "feed": "feed_tokens_space"})
    eng_feeds.to_parquet('eng_feeds')

    # Raw string version of feeds
    eng_feeds['feed_string'] = eng_feeds['feed_tokens_space'].apply(lambda x: ' '.join(x))

    # List-of-comments version of feeds
    eng_feeds['comment_word_indices'] = eng_feeds['comment_lengths'].apply(create_comment_word_indices)

    eng_feeds['feed_comment_list'] = ""
    for index, row in eng_feeds.iterrows():
        comm_w_indices_temp = row['comment_word_indices']
        feed_tokens_space_temp = row['feed_tokens_space']
        inner_list = []
        for i in range(len(comm_w_indices_temp)):
            if i == 0:
                inner_list.append(feed_tokens_space_temp[0:comm_w_indices_temp[i]])
            else:
                inner_list.append(feed_tokens_space_temp[comm_w_indices_temp[i-1]:comm_w_indices_temp[i]])
        eng_feeds.at[index,'feed_comment_list'] = inner_list
    eng_feeds = eng_feeds.drop('comment_word_indices', axis=1)

    # List-of-comments w/ punctuation stripped and lowercase applied version of feeds
    eng_feeds['feed_comment_list_nopunc_lower'] = eng_feeds['feed_comment_list'].apply(strip_punc_and_lower_nested_list)

    # List-of-comments Spacy-tokenized version of feeds
    tokenizer_wrapper(eng_feeds, 'feed_comment_list')

    #eng_feeds.to_pickle("eng_development_feeds_pre_split.pkl")
    #eng_feeds.head(5)

    ## Encode author and proficiency levels, Split into Three Development Cohorts

    # eng_feeds = pd.read_pickle("eng_development_feeds_pre_split.pkl")

    ## Encode author and proficiency as numbers
    labelencoder = LabelEncoder()
    eng_feeds = eng_feeds.reset_index()
    t = eng_feeds['author']
    t = labelencoder.fit_transform(t)
    eng_feeds['author_id'] = t.tolist()
    t = eng_feeds['proficiency']
    t = labelencoder.fit_transform(t)
    eng_feeds['proficiency_id'] = t.tolist()

    ## Split 60 authors into three cohorts
    # Cohort 1: 15 native + 15 non-native
    native_authors = list(set(eng_feeds[eng_feeds['proficiency'] == "N"]['author_id'].values))
    nonnat_authors = list(set(eng_feeds[eng_feeds['proficiency'] == "L"]['author_id'].values))
    all_cohort_native_subset = sample(native_authors, int(n_native / 2))
    all_cohort_nonnat_subset = sample(nonnat_authors, int(n_nonnat / 2))

    cohort_all = pd.concat([eng_feeds[eng_feeds['author_id'].isin(all_cohort_native_subset)], eng_feeds[eng_feeds['author_id'].isin(all_cohort_nonnat_subset)]], ignore_index=False, axis=0)

    # Cohort 2: n_native native
    cohort_native = eng_feeds[eng_feeds['proficiency'] == "N"]

    # Cohort 3: n_nonnat non-native
    cohort_nonnat = eng_feeds[eng_feeds['proficiency'] == "L"]
    
    return cohort_all,cohort_native,cohort_nonnat


def extract_features(cohort,config,filetag):
    # Set parameters according to config (dictionnary)
    n_letter_1gram = config['n_letter_1gram']
    n_letter_2gram = config['n_letter_2gram']
    n_letter_3gram = config['n_letter_3gram']
    n_letter_4gram = config['n_letter_4gram']
    n_digit_1gram  = config['n_digit_1gram']
    #n_digit_2gram = config['n_digit_2gram']
    #n_digit_3gram = config['n_digit_3gram']
    n_punctuation_1gram = config['n_punctuation_1gram']
    n_punctuation_2gram = config['n_punctuation_2gram']
    #n_punctuation_3gram = config['n_punctuation_3gram']
    n_word_1gram = config['n_word_1gram']
    n_word_2gram = config['n_word_2gram']
    n_POS_tag_1gram = config['n_POS_tag_1gram']
    n_POS_tag_2gram = config['n_POS_tag_2gram']
    #n_POS_tag_3gram = config['n_POS_tag_3gram']

    seed = config['seed']

    y = cohort[['author_id', 'intra_author_feed_id']]
    X_train, X_test, y_train, y_test = train_test_split(cohort, y, test_size=0.10, stratify = y['author_id'], random_state=seed)

    for stage in ["train", "test"]:
        if stage == "train":
            feeds_aug = X_train
        elif stage == "test":
            feeds_aug = X_test
            
        # Number of Comments, Median comment length

        feeds_aug['num_comments'] = feeds_aug['comment_lengths'].apply(len)
        feeds_aug['comment_length_median'] = feeds_aug['comment_lengths'].apply(statistics.median)
        feeds_aug = feeds_aug.drop('comment_lengths', axis=1)

        # Character Count, Alphabet Count & Proportion, Digit Count & Proportion, Punctuation Count & Proportion

        feeds_aug = character_count_proportion_wrapper(feeds_aug, 'feed_string', 'letter')
        feeds_aug = character_count_proportion_wrapper(feeds_aug, 'feed_string', 'digit')
        feeds_aug = character_count_proportion_wrapper(feeds_aug, 'feed_string', 'punctuation')
        feeds_aug = character_count_proportion_wrapper(feeds_aug, 'feed_string', 'whitespace')
        feeds_aug = character_count_proportion_wrapper(feeds_aug, 'feed_string', 'character')
    
        # Word Count, Average Word Length, Word Length Distribution (Freq of words length 1-20 letters), Word Case Distribution (All lowercase / First-upper-rest-lowercase / All uppercase / Other), Character case distribution (lowercase / uppercase)

        word_count_wrapper(feeds_aug, 'feed_string')
        word_length_avg_wrapper(feeds_aug, 'feed_string')
        word_length_distribution_wrapper(feeds_aug, 'feed_string')
        word_short_prop_wrapper(feeds_aug, 'word_length_distribution', 'word_count')
        letter_case_distribution_wrapper(feeds_aug, 'feed_string')
        word_case_distribution_wrapper(feeds_aug, 'feed_string')

        # Misspellings Prop

        misspelled_prop_wrapper(feeds_aug, 'feed_string', 'misspelled_prop')
    
        #Stop Word proportion of Tokens
        stop_words_proportion_wrapper(feeds_aug, 'feed_comment_list_spacy', 'stop_words_proportion')
    
        # Vocabulary Richness: Hapax Legomena Proportion of Total Tokens, Hapax Legomena Proportion of Unique Tokens, Unique Tokens over Total Tokens
        # # https://eprints.qut.edu.au/8019/1/8019.pdf
    
        hapax_legomena_proportion_wrapper(feeds_aug, 'feed_comment_list_spacy', 'hapax_legomena_prop_tot_tokens', 'total') # Note: ignores stop words
        hapax_legomena_proportion_wrapper(feeds_aug, 'feed_comment_list_spacy', 'hapax_legomena_prop_unique_tokens', 'unique') # Note: ignores stop words
        token_type_ratio_wrapper(feeds_aug, 'feed_comment_list_spacy', 'token_type_ratio') # Note: ignores stop words

        if stage == "train":
        
            # Letter, Digit, and Punctuation n-grams
            letter_1gram_collection_fromtrain = character_ngrams_wrapper(feeds_aug, 'feed_tokens_space', 'letter_1gram', 1, n_letter_1gram, 'letter')
            letter_2gram_collection_fromtrain = character_ngrams_wrapper(feeds_aug, 'feed_tokens_space', 'letter_2gram', 2, n_letter_2gram, 'letter')
            letter_3gram_collection_fromtrain = character_ngrams_wrapper(feeds_aug, 'feed_tokens_space', 'letter_3gram', 3, n_letter_3gram, 'letter')
            #letter_4gram_collection_fromtrain = character_ngrams_wrapper(feeds_aug, 'feed_tokens_space', 'letter_4gram', 4, n_letter_4gram, 'letter')

            digit_1gram_collection_fromtrain = character_ngrams_wrapper(feeds_aug, 'feed_tokens_space', 'digit_1gram', 1, n_digit_1gram, 'digit')
            #digit_2gram_collection_fromtrain = character_ngrams_wrapper(feeds_aug, 'feed_tokens_space', 'digit_2gram', 2, n_digit_2gram, 'digit')
            #digit_3gram_collection_fromtrain = character_ngrams_wrapper(feeds_aug, 'feed_tokens_space', 'digit_3gram', 3, n_digit_3gram, 'digit')

            punctuation_1gram_collection_fromtrain = character_ngrams_wrapper(feeds_aug, 'feed_tokens_space', 'punctuation_1gram', 1, n_punctuation_1gram, 'punctuation')
            punctuation_2gram_collection_fromtrain = character_ngrams_wrapper(feeds_aug, 'feed_tokens_space', 'punctuation_2gram', 2, n_punctuation_2gram, 'punctuation')
            #punctuation_3gram_collection_fromtrain = character_ngrams_wrapper(feeds_aug, 'feed_tokens_space', 'punctuation_3gram', 3,n_punctuation_3gram, 'punctuation')

            # Word ngrams
            word_1gram_collection_fromtrain = word_ngrams_wrapper(feeds_aug, 'feed_comment_list_nopunc_lower', 'word_1gram', 1, n_word_1gram)
            word_2gram_collection_fromtrain = word_ngrams_wrapper(feeds_aug, 'feed_comment_list_nopunc_lower', 'word_2gram', 2, n_word_2gram)

            # POS n-grams
            POS_tags_1gram_collection_fromtrain = POS_tags_ngram_wrapper(feeds_aug, 'feed_comment_list_spacy', 'POS_tag_1gram', 1, n_POS_tag_1gram)
            POS_tags_2gram_collection_fromtrain = POS_tags_ngram_wrapper(feeds_aug, 'feed_comment_list_spacy', 'POS_tag_2gram', 2, n_POS_tag_2gram)
            #POS_tags_3gram_collection_fromtrain = POS_tags_ngram_wrapper(feeds_aug, 'feed_comment_list_spacy', 'POS_tag_3gram', 3, n_POS_tag_3gram)
            
            # compute the maximum and the minimum values on the train set to perform a min-max scaling later
            max_word_avg =  feeds_aug['word_length_avg'].max()
            max_length_med =  feeds_aug['comment_length_median'].max()
            min_word_avg =  feeds_aug['word_length_avg'].min()
            min_length_med =  feeds_aug['comment_length_median'].min()


        elif stage == "test":
        
            # Letter, Digit, and Punctuation n-grams
            character_ngrams_wrapper(feeds_aug, 'feed_tokens_space', 'letter_1gram', 1, n_letter_1gram, 'letter', letter_1gram_collection_fromtrain)
            character_ngrams_wrapper(feeds_aug, 'feed_tokens_space', 'letter_2gram', 2, n_letter_2gram, 'letter', letter_2gram_collection_fromtrain)
            character_ngrams_wrapper(feeds_aug, 'feed_tokens_space', 'letter_3gram', 3, n_letter_3gram, 'letter', letter_3gram_collection_fromtrain)
            #character_ngrams_wrapper(feeds_aug, 'feed_tokens_space', 'letter_4gram', 4, n_letter_4gram, 'letter', letter_4gram_collection_fromtrain)

            character_ngrams_wrapper(feeds_aug, 'feed_tokens_space', 'digit_1gram', 1, n_digit_1gram, 'digit', digit_1gram_collection_fromtrain)
            #character_ngrams_wrapper(feeds_aug, 'feed_tokens_space', 'digit_2gram', 2, n_digit_2gram, 'digit', digit_2gram_collection_fromtrain)
            #character_ngrams_wrapper(feeds_aug, 'feed_tokens_space', 'digit_3gram', 3, n_digit_3gram, 'digit', digit_3gram_collection_fromtrain)

            character_ngrams_wrapper(feeds_aug, 'feed_tokens_space', 'punctuation_1gram', 1, n_punctuation_1gram, 'punctuation', punctuation_1gram_collection_fromtrain)
            character_ngrams_wrapper(feeds_aug, 'feed_tokens_space', 'punctuation_2gram', 2, n_punctuation_2gram, 'punctuation', punctuation_2gram_collection_fromtrain)
            #character_ngrams_wrapper(feeds_aug, 'feed_tokens_space', 'punctuation_3gram', 3, n_punctuation_3gram, 'punctuation', punctuation_3gram_collection_fromtrain)

            # Word ngrams
            word_ngrams_wrapper(feeds_aug, 'feed_comment_list_nopunc_lower', 'word_1gram', 1, n_word_1gram, word_1gram_collection_fromtrain)
            word_ngrams_wrapper(feeds_aug, 'feed_comment_list_nopunc_lower', 'word_2gram', 2, n_word_2gram, word_2gram_collection_fromtrain)

            # POS n-grams
            POS_tags_ngram_wrapper(feeds_aug, 'feed_comment_list_spacy', 'POS_tag_1gram', 1, n_POS_tag_1gram, POS_tags_1gram_collection_fromtrain)
            POS_tags_ngram_wrapper(feeds_aug, 'feed_comment_list_spacy', 'POS_tag_2gram', 2, n_POS_tag_2gram, POS_tags_2gram_collection_fromtrain)
            #POS_tags_ngram_wrapper(feeds_aug, 'feed_comment_list_spacy', 'POS_tag_3gram', 3, n_POS_tag_3gram, POS_tags_3gram_collection_fromtrain)

        # Perform a min-max normalization using the parameter from the train set
        feeds_aug['word_length_avg'] = (feeds_aug['word_length_avg']-min_word_avg)/(max_word_avg - min_word_avg)
        feeds_aug['comment_length_median'] = (feeds_aug['comment_length_median'] - min_length_med)/(max_length_med - min_length_med)
        
        
        # IMPORTANT: If any features are commented-in above, they must be added to the feature list in this next line
        # Current state : I removed 'letter_4gram',
        feeds_aug = feeds_aug[['proficiency', 'comment_length_median', 'letter_prop', 'digit_prop', 'punctuation_prop', 'whitespace_prop', 'word_length_avg', 'word_length_distribution', 'word_short_prop', 'letter_case_distribution', 'word_case_distribution', 'misspelled_prop', 'stop_words_proportion', 'hapax_legomena_prop_tot_tokens', 'hapax_legomena_prop_unique_tokens', 'token_type_ratio', 'letter_1gram', 'letter_2gram', 'letter_3gram','digit_1gram', 'punctuation_1gram', 'punctuation_2gram', 'word_1gram', 'word_2gram', 'POS_tag_1gram', 'POS_tag_2gram']]
        for col in feeds_aug.columns:
            if type(feeds_aug[col].iloc[0]) == list:
                newcols = [col + "_" + str(i) for i in range(1,len(feeds_aug[col].iloc[0]) + 1)]
                feeds_aug[newcols] = pd.DataFrame(feeds_aug[col].tolist(), index= feeds_aug.index)
                feeds_aug = feeds_aug.drop([col], axis = 1)
            elif type(feeds_aug[col].iloc[0]) == str and col != 'proficiency':
                feeds_aug[col] = pd.to_numeric(feeds_aug[col], downcast="float")
            
        if stage == "train":
            X_train = feeds_aug
        elif stage == "test":
            X_test = feeds_aug
    print("Feature matrix X_train of shape", np.shape(X_train), " has been built")
    
    X_train.to_parquet("X_train"+filetag)
    X_test.to_parquet("X_test"+filetag)
    y_train.to_parquet("y_train"+filetag)
    y_test.to_parquet("y_test"+filetag)
    
    return X_train,X_test,y_train,y_test
    #return letter_1gram_collection_fromtrain, letter_2gram_collection_fromtrain,letter_3gram_collection_fromtrain,letter_4gram_collection_fromtrain,digit_1gram_collection_fromtrain,digit_2gram_collection_fromtrain, digit_3gram_collection_fromtrain,punctuation_1gram_collection_fromtrain, punctuation_2gram_collection_fromtrain,punctuation_3gram_collection_fromtrain,word_1gram_collection_fromtrain,word_2gram_collection_fromtrain,POS_tags_1gram_collection_fromtrain, POS_tags_2gram_collection_fromtrain, POS_tags_3gram_collection_fromtrain

def classify(filetag, kernel,config,X_train,X_test,y_train,y_test):

    # Set SVM parameters 
    degree_svm = config['degree_svm']
    C_svm = config['C_svm']

    train_proficiency = X_train['proficiency']
    X_train = X_train.drop(['proficiency'], axis = 1)#
    test_proficiency = X_test['proficiency']
    X_test = X_test.drop(['proficiency'], axis = 1)

    # Train SVM models - separate if statements because gamma and C can differ. These numbers simply taken from article; could be tuned.
    if kernel == "rbf":
        model = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X_train, y_train['author_id'])
    if kernel == "poly":
        model = svm.SVC(kernel='poly', degree=3, C=1).fit(X_train, y_train['author_id'])
    if kernel == "linear":
        model = svm.SVC(kernel='linear', degree=degree_svm, C=C_svm).fit(X_train, y_train['author_id'])
    
    # Split test into native and nonnative feeds (can handle all-native or all-non-native set)
    X_test_all_proficiencies = X_test.merge(test_proficiency, left_index=True, right_index=True, how='inner')

    X_test_native = X_test_all_proficiencies.loc[X_test_all_proficiencies['proficiency'] == "N"]
    y_test_native = y_test['author_id'].loc[X_test_all_proficiencies['proficiency'] == "N"]
    X_test_native = X_test_native.drop(['proficiency'], axis = 1)
    
    X_test_nonnat = X_test_all_proficiencies.loc[X_test_all_proficiencies['proficiency'] == "L"]
    y_test_nonnat = y_test['author_id'].loc[X_test_all_proficiencies['proficiency'] == "L"]
    X_test_nonnat = X_test_nonnat.drop(['proficiency'], axis = 1)

    # Predict labels
    pred_all = model.predict(X_test)
    # Compute accuracy and f1 score
    accuracy_all = accuracy_score(y_test['author_id'], pred_all)
    f1_all = f1_score(y_test['author_id'], pred_all, average='weighted')
        
    if X_test_native.shape[0] > 0:
        print('Computing accuracy and f1 score on native test authors' )
        pred_native = model.predict(X_test_native)
        accuracy_native = accuracy_score(y_test_native, pred_native)
        f1_native = f1_score(y_test_native, pred_native, average='weighted')
    else:
        accuracy_native = 0
        f1_native = 0
        
    if X_test_nonnat.shape[0] > 0:
        print('Computing accuracy and f1 score on non-native test authors' )
        pred_nonnat = model.predict(X_test_nonnat)
        accuracy_nonnat = accuracy_score(y_test_nonnat, pred_nonnat)
        f1_nonnat = f1_score(y_test_nonnat, pred_nonnat, average='weighted')
    else:
        accuracy_nonnat = 0
        f1_nonnat = 0
    
    # Write results in a yaml file 
    name_dict = 'Model '+ filetag
    dict_file = {name_dict:{'Accuracy' : {'Overall': "%.2f" % (accuracy_all*100),'Native' : "%.2f" % (accuracy_native*100),'Non_Native' : "%.2f" % (accuracy_nonnat*100)},
             'F1': {'Overall':"%.2f" % (f1_all*100),'Native': "%.2f" % (f1_native*100),'Non_Native' :"%.2f" % (f1_nonnat*100)},
             'Infos': {'Model_Kernel' : model.kernel,'Nb_Natives': str(len(set(y_test_native))),'Nb_Non_Native': str(len(set(y_test_nonnat))),'Nb_Feeds_train': str(train_proficiency.shape[0]),'Nb_Feeds_test': str(test_proficiency.shape[0])}}}
    
    return dict_file
    

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
        uptoN_collection = []
        for i in range(upper):
            uptoN_collection.append(c[i][0])
        uptoN_collection.sort()  
    if train_collection != None:
        print("Performing test " + kind + " " + str(n) + "-gram...")
        uptoN_collection = train_collection
    print(uptoN_collection)
    dataframe[newcolumn] = dataframe[feed_token_space].progress_apply(character_ngrams, args = (uptoN_collection,n))
    # Return uptoN_collection for later use with test set - unless this function WAS being called on the test set!
    if train_collection == None:   
        print("Performed train " + kind + " " + str(n) + "-gram in " + str(time.time() - baseline) + " seconds")
        print("Returned up to N most common " + kind + " " + str(n) + "-grams for feature extraction on test set.")
        return uptoN_collection
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

def word_ngrams_wrapper(dataframe, feed_comment_list_nopunc_lower, newcolumn, n, collection_size,train_collection = None):
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
        uptoN_collection = []
        for i in range(upper):
            uptoN_collection.append(c[i][0])
        uptoN_collection.sort() 
    if train_collection != None:
        print("Performing test word " + str(n) + "-gram...")
        uptoN_collection = train_collection
    print(uptoN_collection)
    dataframe[newcolumn] = dataframe[feed_comment_list_nopunc_lower].progress_apply(multchar_ngrams, args = (uptoN_collection,n))
    # Return uptoN_collection for later use with test set - unless this function WAS being called on the test set!
    if train_collection == None:   
        print("Performed train word " + str(n) + "-gram in " + str(time.time() - baseline) + " seconds")
        print("Returned up to N most common word" + str(n) + "-grams for feature extraction on test set.")
        return uptoN_collection
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

def word_length_avg_wrapper(dataframe, feed_string):
    print("Performing word length avg...")
    baseline = time.time()
    dataframe['word_length_avg'] = dataframe[feed_string].apply(lambda x: np.mean([len(w) for w in x.split()]))
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

def POS_tags_ngram_wrapper(dataframe, feed_comment_list_spacy, newcolumn, n, collection_size,train_collection = None):
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
        uptoN_collection = []
        for i in range(upper):
            uptoN_collection.append(c[i][0])
        uptoN_collection.sort() 
    if train_collection != None:
        print("Performing test POS tags " + str(n) + "-gram...")
        dataframe[feed_comment_list_spacy + "_tags"] = dataframe[feed_comment_list_spacy].apply(feed_comment_list_spacy_tags)
        uptoN_collection = train_collection
    print(uptoN_collection)
    dataframe[newcolumn] = dataframe[feed_comment_list_spacy + "_tags"].progress_apply(multchar_ngrams, args = (uptoN_collection,n))
    # Return uptoN_collection for later use with test set - unless this function WAS being called on the test set!
    dataframe = dataframe.drop(feed_comment_list_spacy + "_tags", axis=1)
    if train_collection == None:   
        print("Performed train POS tags " + str(n) + "-gram in " + str(time.time() - baseline) + " seconds")
        print("Returned up to N most common POS tags " + str(n) + "-grams for feature extraction on test set.")
        return uptoN_collection
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