import warnings
warnings.filterwarnings("ignore")
from grid_search_features_functions import *
import argparse
import yaml
from yaml.loader import SafeLoader
import pandas as pd

## Parse command line arguments to know which input file to read and which output file to create
parser = argparse.ArgumentParser()
parser.add_argument("Ntrial", help="read data for trial i")
args = parser.parse_args()

filetag = 's'+str(args.Ntrial)
print("Trial number ",filetag)

## Read config file :
with open('Inputs/After_Tuning/Input_max.yaml', 'r') as f:
    config_max = list(yaml.load_all(f, Loader=SafeLoader))
config_max = config_max[0]


cohort_all = pd.read_pickle('dataset/Test/'+filetag+'_cohort_all')
cohort_native = pd.read_pickle('dataset/Test/'+filetag+'_cohort_native')
cohort_nonnat = pd.read_pickle('dataset/Test/'+filetag+'_cohort_nonnat')
print('extract for cohort all :')
filetag_ = 'all_'+filetag
X_train_all,X_test_all,y_train_all,y_test_all = extract_features(cohort_all,config_max,filetag_)
print('extract for cohort native :')
filetag_ = 'native_'+filetag
X_train_native,X_test_native,y_train_native,y_test_native = extract_features(cohort_native,config_max,filetag_)
print('extract for cohort nonnat : ')
filetag_ = 'nonnat_'+filetag
X_train_nonnat,X_test_nonnat,y_train_nonnat,y_test_nonnat = extract_features(cohort_nonnat,config_max,filetag_)

