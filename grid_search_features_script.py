import warnings
warnings.filterwarnings("ignore")
from grid_search_features_functions import *
import argparse
import yaml
from yaml.loader import SafeLoader

## Parse command line arguments to know which input file to read and which output file to create
parser = argparse.ArgumentParser()
parser.add_argument("Ninput", help="read input file n°Ninput")
args = parser.parse_args()

## Load train/test data 
path_to_dataset = 'dataset/Tunning/'
X_train_all = pd.read_parquet(path_to_dataset+'X_train_cohort_all_500',engine='fastparquet')
X_test_all = pd.read_parquet(path_to_dataset+'X_test_cohort_all_500',engine='fastparquet')
y_train_all = pd.read_parquet(path_to_dataset+'y_train_cohort_all_500',engine='fastparquet')
y_test_all = pd.read_parquet(path_to_dataset+'y_test_cohort_all_500',engine='fastparquet')
X_train_native = pd.read_parquet(path_to_dataset+'X_train_cohort_native_500',engine='fastparquet')
X_test_native = pd.read_parquet(path_to_dataset+'X_test_cohort_native_500',engine='fastparquet')
y_train_native = pd.read_parquet(path_to_dataset+'y_train_cohort_native_500',engine='fastparquet')
y_test_native = pd.read_parquet(path_to_dataset+'y_test_cohort_native_500',engine='fastparquet')
X_train_nonnat = pd.read_parquet(path_to_dataset+'X_train_cohort_nonnat_500',engine='fastparquet')
X_test_nonnat = pd.read_parquet(path_to_dataset+'X_test_cohort_nonnat_500',engine='fastparquet')
y_train_nonnat = pd.read_parquet(path_to_dataset+'y_train_cohort_nonnat_500',engine='fastparquet')
y_test_nonnat = pd.read_parquet(path_to_dataset+'y_test_cohort_nonnat_500',engine='fastparquet')


for i in range(31995, 32000):
        path_to_input_file = 'Inputs/Grid_Search_Features/Input'+str(i)+'.yaml'
        path_to_output_file = 'Outputs/Grid_Search_Features/Output_Features_'+str(i)+'.yaml'

        ## Read config file :
        with open(path_to_input_file, 'r') as f:
                config = list(yaml.load_all(f, Loader=SafeLoader))
        config = config[0]

        result_all = classify_grid_search('cohort_all',config,X_train_all,X_test_all,y_train_all,y_test_all)
        result_native = classify_grid_search('cohort_native',config,X_train_native,X_test_native,y_train_native,y_test_native)
        result_nonnat = classify_grid_search('cohort_nonnat',config,X_train_nonnat,X_test_nonnat,y_train_nonnat,y_test_nonnat)

        with open(path_to_output_file, 'w') as file:
                yaml.dump(result_all, file)
        with open(path_to_output_file, 'a') as file:
                yaml.dump(result_native,file)
                yaml.dump(result_nonnat, file)
