from features_and_svm_functions import *

def change_one_hyperparameter(filetag,config,X_train,X_test,y_train,y_test,var):     
    # Drop feature with current hyperparameter tuning from both
    n_collection = config['n_' + var] # value associated to var        
    columns_to_drop = [var+'_'+str(i) for i in range(n_collection+1,201)]
    X_train_ = X_train.drop(columns = columns_to_drop)
    X_test_ = X_test.drop(columns = columns_to_drop)
    # Re-calculate feature in both with new collection size
    #if var == "letter_1gram":
    #    letter_1gram_collection_fromtrain = character_ngrams_wrapper(X_train, 'feed_tokens_space', 'letter_1gram', 1, n_collection, 'letter')
    #    character_ngrams_wrapper(X_test, 'feed_tokens_space', 'letter_1gram', 1, n_collection, 'letter', letter_1gram_collection_fromtrain)
    #if var == "letter_2gram":
    #    letter_2gram_collection_fromtrain = character_ngrams_wrapper(X_train, 'feed_tokens_space', 'letter_2gram', 2, n_collection, 'letter')
    #    character_ngrams_wrapper(X_test, 'feed_tokens_space', 'letter_2gram', 2, n_collection, 'letter', letter_2gram_collection_fromtrain)
    #if var == "letter_3gram":
    #    letter_3gram_collection_fromtrain = character_ngrams_wrapper(X_train, 'feed_tokens_space', 'letter_3gram', 3, n_collection, 'letter')
    #    character_ngrams_wrapper(X_test, 'feed_tokens_space', 'letter_3gram', 3, n_collection, 'letter', letter_3gram_collection_fromtrain)
    #if var == "digit_1gram":
    #    digit_1gram_collection_fromtrain = character_ngrams_wrapper(X_train, 'feed_tokens_space', 'digit_1gram', 1, n_collection, 'digit')
    #    character_ngrams_wrapper(X_test, 'feed_tokens_space', 'digit_1gram', 1, n_collection, 'digit', digit_1gram_collection_fromtrain)
    #if var == "punctuation_1gram":
    #    punctuation_1gram_collection_fromtrain = character_ngrams_wrapper(X_train, 'feed_tokens_space', 'punctuation_1gram', 1, n_collection, 'punctuation')
    #    character_ngrams_wrapper(X_test, 'feed_tokens_space', 'punctuation_1gram', 1, n_collection, 'punctuation', punctuation_1gram_collection_fromtrain)
    #if var == "punctuation_2gram":
    #    punctuation_2gram_collection_fromtrain = character_ngrams_wrapper(X_train, 'feed_tokens_space', 'punctuation_2gram', 2, n_collection, 'punctuation')
    #    character_ngrams_wrapper(X_test, 'feed_tokens_space', 'punctuation_2gram', 2, n_collection, 'punctuation', punctuation_2gram_collection_fromtrain)
    #if var == "word_1gram":
    #    word_1gram_collection_fromtrain = word_ngrams_wrapper(X_train, 'feed_comment_list_nopunc_lower', 'word_1gram', 1, n_collection)
    #    word_ngrams_wrapper(X_test, 'feed_comment_list_nopunc_lower', 'word_1gram', 1, n_collection, word_1gram_collection_fromtrain)
    #if var == "word_2gram":
    #    word_2gram_collection_fromtrain = word_ngrams_wrapper(X_train, 'feed_comment_list_nopunc_lower', 'word_2gram', 2, n_collection)
    #    word_ngrams_wrapper(X_test, 'feed_comment_list_nopunc_lower', 'word_2gram', 2, n_collection, word_2gram_collection_fromtrain)
    #if var == "POS_tag_1gram":
    #    POS_tags_1gram_collection_fromtrain = POS_tags_ngram_wrapper(X_train, 'feed_comment_list_spacy', 'POS_tag_1gram', 1, n_collection)
    #    POS_tags_ngram_wrapper(X_test, 'feed_comment_list_spacy', 'POS_tag_1gram', 1, n_collection, POS_tags_1gram_collection_fromtrain)
    #if var == "POS_tag_2gram":
    #    POS_tags_2gram_collection_fromtrain = POS_tags_ngram_wrapper(X_train, 'feed_comment_list_spacy', 'POS_tag_2gram', 2, n_collection)
    #    POS_tags_ngram_wrapper(X_test, 'feed_comment_list_spacy', 'POS_tag_2gram', 2, n_collection, POS_tags_2gram_collection_fromtrain)
    #        
    # Beautify feature sets to prepare for classification
    # IMPORTANT: If any features are added, they must be added to the feature list in this next line
    #keep_vars = ['proficiency', 'comment_length_median', 'letter_prop', 'digit_prop', 'punctuation_prop', 'whitespace_prop', 'word_length_avg', 'word_length_distribution', 'word_short_prop', 'letter_case_distribution', 'word_case_distribution', 'misspelled_prop', 'stop_words_proportion', 'hapax_legomena_prop_tot_tokens', 'hapax_legomena_prop_unique_tokens', 'token_type_ratio', 'letter_1gram', 'letter_2gram', 'letter_3gram', 'digit_1gram', 'punctuation_1gram', 'punctuation_2gram', 'word_1gram', 'word_2gram', 'POS_tag_1gram', 'POS_tag_2gram']
    #X_train = X_train[keep_vars]
    #for col in X_train.columns:
    #    if type(X_train[col].iloc[0]) == list:
    #        newcols = [col + "_" + str(i) for i in range(1,len(X_train[col].iloc[0]) + 1)]
    #        X_train[newcols] = pd.DataFrame(X_train[col].tolist(), index= X_train.index)
    #        X_train = X_train.drop([col], axis = 1)
    #    elif type(X_train[col].iloc[0]) == str and col != 'proficiency':
    #        X_train[col] = pd.to_numeric(X_train[col], downcast="float")
    #    X_test = X_test[keep_vars]
    #    for col in X_test.columns:
    #        if type(X_test[col].iloc[0]) == list:
    #            newcols = [col + "_" + str(i) for i in range(1,len(X_test[col].iloc[0]) + 1)]
    #            X_test[newcols] = pd.DataFrame(X_test[col].tolist(), index= X_test.index)
    #            X_test = X_test.drop([col], axis = 1)
    #        elif type(X_test[col].iloc[0]) == str and col != 'proficiency':
    #            X_test[col] = pd.to_numeric(X_test[col], downcast="float")
                        
    # Overwrite X_train and X_test used in classify function (baseline set still saved at dev_ + filetag + _X_train/test_ugly_features.pkl
    #X_train.to_pickle("dev_" + filetag + "_X_train.pkl")
    #X_test.to_pickle("dev_" + filetag + "_X_test.pkl")
            
    # Results with current n-gram collection size

    #print("~~~~~")
    print("PERFORMING SVM FOR " + filetag + " WITH UPDATED " + var + ", " + str(n_collection) + " N-GRAMS IN THE COLLECTION")
    dict_result = classify(filetag, "linear",config,X_train_,X_test_,y_train,y_test)
    #print("~~~~~")
    return dict_result      