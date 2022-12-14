import argparse
import pickle5 as pickle
from features_and_svm_functions import *
import warnings
import yaml # for the config and output files
from yaml.loader import SafeLoader

## Parse command line arguments to know which input file to read and which output file to create
parser = argparse.ArgumentParser()
parser.add_argument("Ninput", help="read input file n°Ninput")
args = parser.parse_args()
path_to_input_file = 'Inputs/input'+args.Ninput+'.yaml'
path_to_output_file = 'Outputs/output'+args.Ninput+'.yaml'

## Read config file :
with open(path_to_input_file, 'r') as f:
    config = list(yaml.load_all(f, Loader=SafeLoader))
config = config[0]

## Load data
eng_native = pd.read_pickle('dataset/Tunning/30native_english')
eng_nonnat = pd.read_pickle('dataset/Tunning/30non_native_english')
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

result_all = classify("cohort_all", "linear",cohort_all,config,path_to_output_file)
result_native = classify("cohort_native", "linear",cohort_native,config,path_to_output_file)   
result_nonnat = classify("cohort_nonnat", "linear",cohort_nonnat,config, path_to_output_file)

with open(path_to_output_file, 'w') as file:
        yaml.dump(result_all, file)
with open(path_to_output_file, 'a') as file:
        yaml.dump(result_native,file)
        yaml.dump(result_nonnat)

#####################################################################################################################

