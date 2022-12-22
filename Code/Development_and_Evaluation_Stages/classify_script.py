import warnings
warnings.filterwarnings("ignore")
from features_and_svm_functions import *
import argparse
import yaml
from yaml.loader import SafeLoader

## Parse command line arguments to know which input file to read and which output file to create
parser = argparse.ArgumentParser()
parser.add_argument("Ninput", help="read input file n°Ninput")
args = parser.parse_args()

# Path to input/output folders
path_out = '../../Data/Outputs/Grid_Search_Features/'
path_in =  '../../Data/Inputs/Grid_Search_Features/'   ## This folder has to be filled manually bc too many files (extract from ../../Data/Tuning/Feature Tuning Dataframes/inputs_df)

## Load train/test data 
path_to_dataset = '../../Data/Tuning/'
X_train_all = pd.read_pickle(path_to_dataset + 'X_train_cohort_all_500.pkl')
X_test_all = pd.read_pickle(path_to_dataset + 'X_test_cohort_all_500.pkl')
y_train_all = pd.read_pickle(path_to_dataset + 'y_train_cohort_all_500.pkl')
y_test_all = pd.read_pickle(path_to_dataset + 'y_test_cohort_all_500.pkl')
X_train_native= pd.read_pickle(path_to_dataset + 'X_train_cohort_native_500.pkl')
X_test_native = pd.read_pickle(path_to_dataset + 'X_test_cohort_native_500.pkl')
y_train_native = pd.read_pickle(path_to_dataset + 'y_train_cohort_native_500.pkl')
y_test_native = pd.read_pickle(path_to_dataset + 'y_test_cohort_native_500.pkl')
X_train_nonnat = pd.read_pickle(path_to_dataset + 'X_train_cohort_nonnat_500.pkl')
X_test_nonnat = pd.read_pickle(path_to_dataset + 'X_test_cohort_nonnat_500.pkl')
y_train_nonnat = pd.read_pickle(path_to_dataset + 'y_train_cohort_nonnat_500.pkl')
y_test_nonnat = pd.read_pickle(path_to_dataset + 'y_test_cohort_nonnat_500.pkl')

print('Task ', args.Ninput)

for i in range(int(args.Ninput)*2000, (int(args.Ninput) + 1)*2000):
        
        # Read input file 
        path_to_input_file = path_in + 'Input'+str(i+1)+'.yaml'
        path_to_output_file = path_out + 'Output_Features_'+str(i+1)+'.yaml'
        
        with open(path_to_input_file, 'r') as f:
                config = list(yaml.load_all(f, Loader=SafeLoader))
        config = config[0]
        
        # Train model, classify data, test model
        result_all = classify_with_input('cohort_all',config,X_train_all,X_test_all,y_train_all,y_test_all)
        result_native = classify_with_input('cohort_native',config,X_train_native,X_test_native,y_train_native,y_test_native)
        result_nonnat = classify_with_input('cohort_nonnat',config,X_train_nonnat,X_test_nonnat,y_train_nonnat,y_test_nonnat)
        
        # Write results in output file
        with open(path_to_output_file, 'w') as file:
                yaml.dump(result_all, file)
        with open(path_to_output_file, 'a') as file:
                yaml.dump(result_native,file)
                yaml.dump(result_nonnat, file)
