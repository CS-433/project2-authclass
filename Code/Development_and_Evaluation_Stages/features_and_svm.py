import argparse
from features_and_svm_functions import *
import warnings
import yaml # for the config and output files
from yaml.loader import SafeLoader

#*******************************************************************************************************************
#   This file can be used to perform the whole 'feature extraction + classification process' on the three cohorts
#   Feature extraction takes about 1 hour 30 minutes to be performed on the three cohorts 
#*******************************************************************************************************************

## Parse command line arguments to know which input file to read and which output file to create
parser = argparse.ArgumentParser()
parser.add_argument("Ninput", help="read input file n°Ninput")
args = parser.parse_args()

## Define paths to input and output files
path_to_input_file = '../../Data/Inputs/Input'+args.Ninput+'.yaml'
path_to_output_file = '../../Data/Outputs/Output'+args.Ninput+'.yaml'

## Read config file :
with open(path_to_input_file, 'r') as f:
    config = list(yaml.load_all(f, Loader=SafeLoader))
config = config[0]

## Load data
eng_native = pd.read_parquet('../../Data/Tuning/30native_english',engine='fastparquet')
eng_nonnat = pd.read_parquet('../../Data/Tuning/30non_native_english',engine='fastparquet')
eng_native['proficiency'] = "N" # N = native
eng_nonnat['proficiency'] = "L" # L = learner

## Build three cohorts according to proficiency :
#   - Cohort 1: 15 Native + 15 Non-Native
#   - Cohort 2: 15 Native from Cohort 1 + another random 15 native
#   - Cohort 3: 15 Non-Native from Cohort 2 + another random 15 non-native
#  Therefore, we need 30 native and 30 non-native authors

num_native_authors_to_sample = 30
num_nonnat_authors_to_sample = 30

cohort_all, cohort_native, cohort_nonnat = build_cohorts(num_native_authors_to_sample,
                                                         num_nonnat_authors_to_sample,
                                                         eng_native,
                                                         eng_nonnat,
                                                         config['seed'])

## Extract features for each cohort, train the 3 corresponding models, and test them separately
#  Based on https://www.baeldung.com/cs/svm-multiclass-classification
#  Results (accuracies and f1 scores) dor each model are then written in output file

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

X_train_all,X_test_all,y_train_all,y_test_all= extract_features(cohort_all,config,"cohort_all")
X_train_native,X_test_native,y_train_native,y_test_native= extract_features(cohort_native,config,"cohort_native")
X_train_nonnat,X_test_nonnat,y_train_nonnat,y_test_nonnat= extract_features(cohort_nonnat,config,"cohort_nonnat")

result_all = classify("cohort_all", "linear",config,X_train_all,X_test_all,y_train_all,y_test_all)
result_native = classify("cohort_native", "linear",config,X_train_native,X_test_native,y_train_native,y_test_native)   
result_nonnat = classify("cohort_nonnat", "linear",config,X_train_nonnat,X_test_nonnat,y_train_nonnat,y_test_nonnat)

with open(path_to_output_file, 'w') as file:
        yaml.dump(result_all, file)
with open(path_to_output_file, 'a') as file:
        yaml.dump(result_native,file)
        yaml.dump(result_nonnat, file)

#####################################################################################################################

