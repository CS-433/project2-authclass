# %% [markdown]
# # Sample Authors and Prepare Feeds for Feature Extraction

# %%
# May need to install packages seen in feature_extraction_functions
from feature_extraction_functions import *

team_seed = 13 + 4 + 5
random.seed(team_seed) 

# Space-split, long-format version of feeds (Arthur's version + some cleaning and wide-to-long conversion)
eng_native = pd.read_pickle('Data/Classified/native_english_40feeds')
eng_nonnat = pd.read_pickle('Data/Classified/non_native_english_40feeds')
eng_native['proficiency'] = "N" # native
eng_nonnat['proficiency'] = "L" # learner

# Three cohorts will be created based on development split below.
# Cohort 1: 15 Native + 15 Non-Native
# Cohort 2: 15 Native from Cohort 1 + another random 15 native
# Cohort 3: 15 Non-Native from Cohort 2 + another random 15 non-native
# Therefore, we need 30 native and 30 non-native authors for the development cohorts
num_native_authors_to_sample = 30
num_nonnat_authors_to_sample = 30
eng_native_sample = eng_native.sample(num_native_authors_to_sample, random_state=team_seed)
eng_nonnat_sample = eng_nonnat.sample(num_nonnat_authors_to_sample, random_state=team_seed)

### ARTHUR REMOVE THIS ONCE YOU LIMIT TO 20 FEEDS PER AUTHOR ###
eng_native_sample = eng_native_sample[['timerange', 'file1', 'slices1', 'file2', 'slices2', 'file3', 'slices3',
       'file4', 'slices4', 'file5', 'slices5', 'file6', 'slices6', 'file7',
       'slices7', 'file8', 'slices8', 'file9', 'slices9', 'file10', 'slices10',
       'file11', 'slices11', 'file12', 'slices12', 'file13', 'slices13',
       'file14', 'slices14', 'file15', 'slices15', 'file16', 'slices16',
       'file17', 'slices17', 'file18', 'slices18', 'file19', 'slices19',
       'file20', 'slices20', 'proficiency']]

eng_nonnat_sample = eng_nonnat_sample[['timerange', 'file1', 'slices1', 'file2', 'slices2', 'file3', 'slices3',
       'file4', 'slices4', 'file5', 'slices5', 'file6', 'slices6', 'file7',
       'slices7', 'file8', 'slices8', 'file9', 'slices9', 'file10', 'slices10',
       'file11', 'slices11', 'file12', 'slices12', 'file13', 'slices13',
       'file14', 'slices14', 'file15', 'slices15', 'file16', 'slices16',
       'file17', 'slices17', 'file18', 'slices18', 'file19', 'slices19',
       'file20', 'slices20', 'proficiency']]

# Before splitting into cohorts, perform all pre-feature-extraction processing
eng_feeds = pd.concat([eng_nonnat_sample, eng_native_sample], ignore_index=False, axis=0) # 
eng_feeds['author'] = eng_feeds.index
eng_feeds = pd.wide_to_long(eng_feeds, ["file", "slices"], i="author", j="intra_author_feed_id").sort_index()
eng_feeds = eng_feeds.rename(columns={"slices": "comment_lengths", "file": "feed_tokens_space"})

# Raw string version of feeds
def feed_string(feed):
    return ' '.join(feed)
eng_feeds['feed_string'] = eng_feeds['feed_tokens_space'].apply(feed_string)

# List-of-comments version of feeds
## Doing this via loop because two columns involved in function instead of one...
def create_comment_word_indices(comment_lengths):
    np_comment_lengths = np.array(comment_lengths)
    return np.cumsum(np_comment_lengths)
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
def strip_punc_and_lower_nested_list(feed_comment_list):
    feed_comment_list_nopunc_lower = []
    for comment in feed_comment_list:
        feed_comment_list_nopunc_lower.append(re.sub(r'[^A-Za-z0-9 ]+', '', ' '.join(comment)).lower().split()) 
    return feed_comment_list_nopunc_lower
eng_feeds['feed_comment_list_nopunc_lower'] = eng_feeds['feed_comment_list'].apply(strip_punc_and_lower_nested_list)

# List-of-comments Spacy-tokenized version of feeds
tokenizer_wrapper(eng_feeds, 'feed_comment_list')

eng_feeds.to_pickle("eng_development_feeds_pre_split.pkl")

eng_feeds.head(5)

# %% [markdown]
# # Encode author and proficiency levels, Split into Three Development Cohorts

# %%
from feature_extraction_functions import *
from random import sample
team_seed = 13 + 4 + 5
random.seed(team_seed) 

eng_feeds = pd.read_pickle("eng_development_feeds_pre_split.pkl")

# Encode author and proficiency as numbers
labelencoder = LabelEncoder()
eng_feeds = eng_feeds.reset_index()
t = eng_feeds['author']
t = labelencoder.fit_transform(t)
eng_feeds['author_id'] = t.tolist()
t = eng_feeds['proficiency']
t = labelencoder.fit_transform(t)
eng_feeds['proficiency_id'] = t.tolist()

# %%
# Split 60 authors into three cohorts
# Cohort 1: 15 native + 15 non-native
native_authors = list(set(eng_feeds[eng_feeds['proficiency'] == "N"]['author_id'].values))
nonnat_authors = list(set(eng_feeds[eng_feeds['proficiency'] == "L"]['author_id'].values))
all_cohort_native_subset = sample(native_authors, int(num_native_authors_to_sample / 2))
all_cohort_nonnat_subset = sample(nonnat_authors, int(num_nonnat_authors_to_sample / 2))
cohort_all = pd.concat([eng_feeds[eng_feeds['author_id'].isin(all_cohort_native_subset)], eng_feeds[eng_feeds['author_id'].isin(all_cohort_nonnat_subset)]], ignore_index=False, axis=0)
# Cohort 2: 30 native
cohort_native = eng_feeds[eng_feeds['proficiency'] == "N"]
# Cohort 3: 30 non-native
cohort_nonnat = eng_feeds[eng_feeds['proficiency'] == "L"]

# %% [markdown]
# # Extract Features

# %%
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import yaml # for the config file
from yaml.loader import SafeLoader
pd.options.mode.chained_assignment = None


team_seed = 13 + 4 + 5

# read config file :
with open('config_svm.yaml', 'r') as f:
    config = list(yaml.load_all(f, Loader=SafeLoader))

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
n_POS_tag_3gram = config['n_POS_tag_3gram']

degree_svm = config['degree']
C_svm = config['C']

def extract_features(cohort, filetag):
    
    y = cohort[['author_id', 'intra_author_feed_id']]
    X_train, X_test, y_train, y_test = train_test_split(cohort, y, test_size=0.10, stratify = y['author_id'], random_state=team_seed)

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
        
            global letter_1gram_collection_fromtrain
            global letter_2gram_collection_fromtrain
            global letter_3gram_collection_fromtrain
            global letter_4gram_collection_fromtrain
            global digit_1gram_collection_fromtrain
            global digit_2gram_collection_fromtrain
            global digit_3gram_collection_fromtrain
            global punctuation_1gram_collection_fromtrain
            global punctuation_2gram_collection_fromtrain
            global punctuation_3gram_collection_fromtrain
            global word_1gram_collection_fromtrain
            global word_2gram_collection_fromtrain
            global POS_tags_1gram_collection_fromtrain
            global POS_tags_2gram_collection_fromtrain
            global POS_tags_3gram_collection_fromtrain
        
            # Letter, Digit, and Punctuation n-grams
            letter_1gram_collection_fromtrain = character_ngrams_wrapper(feeds_aug, 'feed_tokens_space', 'letter_1gram', 1, n_letter_1gram, 'letter')
            letter_2gram_collection_fromtrain = character_ngrams_wrapper(feeds_aug, 'feed_tokens_space', 'letter_2gram', 2, n_letter_2gram, 'letter')
            letter_3gram_collection_fromtrain = character_ngrams_wrapper(feeds_aug, 'feed_tokens_space', 'letter_3gram', 3, n_letter_3gram, 'letter')
            letter_4gram_collection_fromtrain = character_ngrams_wrapper(feeds_aug, 'feed_tokens_space', 'letter_4gram', 4, n_letter_4gram, 'letter')

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
            #POS_tags_3gram_collection_fromtrain = POS_tags_ngram_wrapper(feeds_aug, 'feed_comment_list_spacy', 'POS_tag_3gram', 3, 5n_POS_tag_3gram)

        elif stage == "test":
        
            # Letter, Digit, and Punctuation n-grams
            character_ngrams_wrapper(feeds_aug, 'feed_tokens_space', 'letter_1gram', 1, n_letter_1gram, 'letter', letter_1gram_collection_fromtrain)
            character_ngrams_wrapper(feeds_aug, 'feed_tokens_space', 'letter_2gram', 2, n_letter_2gram, 'letter', letter_2gram_collection_fromtrain)
            character_ngrams_wrapper(feeds_aug, 'feed_tokens_space', 'letter_3gram', 3, n_letter_3gram, 'letter', letter_3gram_collection_fromtrain)
            character_ngrams_wrapper(feeds_aug, 'feed_tokens_space', 'letter_4gram', 4, n_letter_4gram, 'letter', letter_4gram_collection_fromtrain)

            character_ngrams_wrapper(feeds_aug, 'feed_tokens_space', 'digit_1gram', 1, n_digit_1gram, 'digit', digit_1gram_collection_fromtrain)
            #character_ngrams_wrapper(feeds_aug, 'feed_tokens_space', 'digit_2gram', 2, n_digit_2gram, 'digit', digit_2gram_collection_fromtrain)
            #character_ngrams_wrapper(feeds_aug, 'feed_tokens_space', 'digit_3gram', 3, n_digit_3gram, 'digit', digit_3gram_collection_fromtrain)

            character_ngrams_wrapper(feeds_aug, 'feed_tokens_space', 'punctuation_1gram', 1, n_punctuation_1gram, 'punctuation', punctuation_1gram_collection_fromtrain)
            character_ngrams_wrapper(feeds_aug, 'feed_tokens_space', 'punctuation_2gram', 2, n_punctuation_2gram, 'punctuation', punctuation_2gram_collection_fromtrain)
            #character_ngrams_wrapper(feeds_aug, 'feed_tokens_space', 'punctuation_3gram', 3, n_punctuation_3gram, 'punctuation', punctuation_3gram_collection_fromtrain)

            # Word ngrams
            word_ngrams_wrapper(feeds_aug, 'feed_comment_list_nopunc_lower', 'word_1gram', 1, n_word_1gram, word_1gram_collection_fromtrain)
            word_ngrams_wrapper(feeds_aug, 'feed_comment_list_nopunc_lower', 'word_1gram', 2, n_word_2gram, word_2gram_collection_fromtrain)

            # POS n-grams
            POS_tags_ngram_wrapper(feeds_aug, 'feed_comment_list_spacy', 'POS_tag_1gram', 1, n_POS_tag_1gram, POS_tags_1gram_collection_fromtrain)
            POS_tags_ngram_wrapper(feeds_aug, 'feed_comment_list_spacy', 'POS_tag_2gram', 2, n_POS_tag_2gram, POS_tags_2gram_collection_fromtrain)
            #POS_tags_ngram_wrapper(feeds_aug, 'feed_comment_list_spacy', 'POS_tag_3gram', 3, n_POS_tag_3gram, POS_tags_3gram_collection_fromtrain)

        # IMPORTANT: If any features are commented-in above, they must be added to the feature list in this next line
        feeds_aug = feeds_aug[['proficiency', 'comment_length_median', 'letter_prop', 'digit_prop', 'punctuation_prop', 'whitespace_prop', 'word_length_avg', 'word_length_distribution', 'word_short_prop', 'letter_case_distribution', 'word_case_distribution', 'misspelled_prop', 'stop_words_proportion', 'hapax_legomena_prop_tot_tokens', 'hapax_legomena_prop_unique_tokens', 'token_type_ratio', 'letter_1gram', 'letter_2gram', 'letter_3gram', 'letter_4gram', 'digit_1gram', 'punctuation_1gram', 'punctuation_2gram', 'word_1gram', 'word_2gram', 'POS_tag_1gram', 'POS_tag_2gram']]
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
        
    X_train.to_pickle("dev_" + filetag + "_X_train.pkl")
    X_test.to_pickle("dev_" + filetag + "_X_test.pkl")
    y_train.to_pickle("dev_" + filetag + "_y_train.pkl")
    y_test.to_pickle("dev_" + filetag + "_y_test.pkl")

extract_features(cohort_all, "cohort_all")
extract_features(cohort_native, "cohort_native")
extract_features(cohort_nonnat, "cohort_nonnat")

# %% [markdown]
# # Train Models

# %%
# https://www.baeldung.com/cs/svm-multiclass-classification

# %%
def classify(filetag, kernel):
    # Import train and test data w features. Split out proficiency.
    X_train = pd.read_pickle('dev_' + filetag + '_X_train.pkl')
    train_proficiency = X_train['proficiency']
    X_train = X_train.drop(['proficiency'], axis = 1)

    X_test = pd.read_pickle('dev_' + filetag + '_X_test.pkl')
    test_proficiency = X_test['proficiency']
    X_test = X_test.drop(['proficiency'], axis = 1)
    
    y_train = pd.read_pickle('dev_' + filetag + '_y_train.pkl')
    y_test = pd.read_pickle('dev_' + filetag + '_y_test.pkl')

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
    
    # Predict native and non-native
    pred_all = model.predict(X_test)
    accuracy_all = accuracy_score(y_test['author_id'], pred_all)
    f1_all = f1_score(y_test['author_id'], pred_all, average='weighted')
        
    if X_test_native.shape[0] > 0:
        pred_native = model.predict(X_test_native)
        accuracy_native = accuracy_score(y_test_native, pred_native)
        f1_native = f1_score(y_test_native, pred_native, average='weighted')
    else:
        accuracy_native = 0
        f1_native = 0
        
    if X_test_nonnat.shape[0] > 0:
        pred_nonnat = model.predict(X_test_nonnat)
        accuracy_nonnat = accuracy_score(y_test_nonnat, pred_nonnat)
        f1_nonnat = f1_score(y_test_nonnat, pred_nonnat, average='weighted')
    else:
        accuracy_nonnat = 0
        f1_nonnat = 0
            
    print('SVM ' + model.kernel + ' kernel results for ' + filetag + ':')
    print(' ')
    print('- Accuracy')
    print('-- Overall: ', "%.2f" % (accuracy_all*100))
    print('-- Native:', "%.2f" % (accuracy_native*100))
    print('-- Non-Native:', "%.2f" % (accuracy_nonnat*100))
    print(' ')
    print('- F1 Score')
    print('-- Overall:', "%.2f" % (f1_all*100))
    print('-- Native:', "%.2f" % (f1_native*100))
    print('-- Non-Native:', "%.2f" % (f1_nonnat*100))
    print(' ')
    print(' ')
    print('Results are for ' + str(len(set(y_test_native))) + ' native ' + str(len(set(y_test_nonnat))) + ' non-native english authors. Model trained on ' + str(train_proficiency.shape[0]) + ' feeds and tested on ' + str(test_proficiency.shape[0]) + ' feeds.') 
    print(' ')
    print(' ')
    print(' ')
    print(' ')
    
classify("cohort_all", "linear")
classify("cohort_native", "linear")
classify("cohort_nonnat", "linear")

#classify("cohort_all", "rbf")
#classify("cohort_native", "rbf")
#classify("cohort_nonnat", "rbf")

#classify("cohort_all", "poly")
#classify("cohort_native", "poly")
#classify("cohort_nonnat", "poly")


