import yaml # for the config file
import numpy as np
from sklearn.model_selection import ParameterGrid
import pandas as pd

penalty                 = np.array([2])
C_svm                   = np.array([3])
seed                    = np.array([22])
letter_1gram_range      = np.array([13,26])
letter_2gram_range      = np.array([20,100,500])
letter_3gram_range      = np.array([20,100,500])
letter_4gram_range      = np.array([20,100,500])
digit_1gram_range       = np.array([5,10])
punctuation_1gram_range = np.array([18,36])
punctuation_2gram_range = np.array([20,100,500])
word_1gram_range        = np.array([20,100,500])
word_2gram_range        = np.array([20,100,500])
POS_tag_1gram_range     = np.array([24,48])
POS_tag_2gram_range     = np.array([20,100,500])
#C_svm_range             = [0.5,1,3,5,10]

listKeys = ['C_svm',
            'n_POS_tag_1gram',
            'n_letter_2gram',
            'n_digit_1gram',
            'n_letter_1gram',
            'n_letter_3gram',
            'n_letter_4gram',
            'n_word_1gram',
            'penalty',
            'n_punctuation_1gram',
            'n_punctuation_2gram',
            'seed',
            'n_word_2gram',
            'n_POS_tag_2gram']

param_grid = {'seed': seed,
              'C_svm':C_svm,
              'penalty':penalty,
              'letter_1gram_range':letter_1gram_range,
              'letter_2gram_range':letter_2gram_range,
              'letter_3gram_range': letter_3gram_range,
              'letter_4gram_range': letter_4gram_range,
              'digit_1gram_range':digit_1gram_range,
              'word_1gram_range':word_1gram_range,
              'word_2gram_range':word_2gram_range,
              'punctuation_1gram_range':punctuation_1gram_range,
              'punctuation_2gram_range':punctuation_2gram_range,
              'POS_tag_1gram_range':POS_tag_1gram_range,
              'POS_tag_2gram_range':POS_tag_2gram_range}

grid = ParameterGrid(param_grid)
id = 0
list_of_dicts = []
for dicts in grid :
    id+=1
    path_to_input_file = 'Inputs/Grid_Search_Features/Input'+str(id)+'.yaml'
    listValues = list(dicts.values())
    arrayValues = np.array(listValues)
    listValues = arrayValues.tolist()
    features_param = {listKeys[i]: listValues[i] for i in range(0, len(listKeys), 1)}
    list_of_dicts.append(features_param)
    with open(path_to_input_file, 'w') as file:
        yaml.dump(features_param, file) 

df = pd.DataFrame(list_of_dicts)
df.to_pickle('Inputs/Grid_Search_Features/inputs_df')