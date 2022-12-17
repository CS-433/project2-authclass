from features_and_svm_functions import *

def find_features_param(dict_keys):
    list_keys = list(dict_keys)
    for s in list_keys:
        if (s == 'C_svm') | (s=='penalty') | (s == 'seed') :
            list_keys.remove(s)
    return list_keys

def max_collection_size(var,X):
    n= len([col for col in X if col.startswith(var)])
    return n

def classify_grid_search(filetag,config,X_train,X_test,y_train,y_test):     
    # Drop feature with current hyperparameter tuning from both
    X_train_ = X_train
    X_test_ = X_test
    for var in find_features_param(config.keys()) :
        n_collection = config[var] # value associated to var = n_xxx        
        var = var[2:] # now we consider the name xxx : remove n_
        columns_to_drop = [var+'_'+str(i) for i in range(n_collection+1,max_collection_size(var,X_train))]
        X_train_ = X_train_.drop(columns = columns_to_drop)
        X_test_ = X_test_.drop(columns = columns_to_drop)
    dict_result = classify(filetag, "linear",config,X_train_,X_test_,y_train,y_test)
    return dict_result