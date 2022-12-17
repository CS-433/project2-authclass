import pandas as pd
import yaml 
from yaml.loader import SafeLoader



output_all = {'acc_native':     [], 
              'acc_non_native': [],
              'acc_overall':    [],
              'f1_native':      [],
              'f1_non_native':  [],
              'f1_overall':     []}

output_native = {'accuracy': [],
                 'f1':       []}

output_non_native = { 'accuracy': [],
                     'f1':        []}

path = 'Outputs/'

for i in range(1,21):
    file = 'Output_HyperPara_' + str(i) + '.yaml'
    with open(path + file, 'r') as f:
        config = list(yaml.load_all(f, Loader=SafeLoader))
    config = config[0]

    output_all['acc_native'].append(config['Model cohort_all']['Accuracy']['Native'])
    output_all['acc_non_native'].append(config['Model cohort_all']['Accuracy']['Non_Native'])
    output_all['acc_overall'].append(config['Model cohort_all']['Accuracy']['Overall'])
    output_all['f1_native'].append(config['Model cohort_all']['F1']['Native'])
    output_all['f1_non_native'].append(config['Model cohort_all']['F1']['Non_Native'])
    output_all['f1_overall'].append(config['Model cohort_all']['F1']['Overall'])

    output_native['accuracy'].append(config['Model cohort_native']['Accuracy']['Native'])
    output_native['f1'].append(config['Model cohort_native']['F1']['Native'])
    
    output_non_native['accuracy'].append(config['Model cohort_nonnat']['Accuracy']['Non_Native'])
    output_non_native['f1'].append(config['Model cohort_nonnat']['F1']['Non_Native'])


output_all = pd.DataFrame(output_all)
output_native = pd.DataFrame(output_native)
output_non_native = pd.DataFrame(output_non_native)

output_all.to_pickle('Data/grid_search_results/output_all.pkl')
output_native.to_pickle('Data/grid_search_results/output_native.pkl')
output_non_native.to_pickle('Data/grid_search_results/output_non_native.pkl')


