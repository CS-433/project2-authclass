from features_and_svm_functions import *

def change_one_parameter(filetag,config,X_train,X_test,y_train,y_test,var):     
    # Drop feature with current hyperparameter tuning from both
    n_collection = config['n_' + var] # value associated to var        
    columns_to_drop = [var+'_'+str(i) for i in range(n_collection+1,201)]
    X_train_ = X_train.drop(columns = columns_to_drop)
    X_test_ = X_test.drop(columns = columns_to_drop)

    print("PERFORMING SVM FOR " + filetag + " WITH UPDATED " + var + ", " + str(n_collection) + " N-GRAMS IN THE COLLECTION")
    dict_result = classify(filetag, "linear",config,X_train_,X_test_,y_train,y_test)
    #print("~~~~~")
    return dict_result