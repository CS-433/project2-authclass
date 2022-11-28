import pandas as pd
import numpy as np
import random as random
import langdetect
from langdetect import DetectorFactory
from tqdm import tqdm

def langdetect_dataframe(df,seed =0):
    ''' Detect the language of each body in the dataframe.  
    ''' 
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        df.at[index,'body_lang']=langdetect_process(row['body'],seed=seed)
    return df

def langdetect_process(comment,seed):
    """ Classification between English 'en', French 'fr', other languages 'N', and undefined 'U' if the classification fails.
        Use the package langdetect https://pypi.org/project/langdetect/
    """
    DetectorFactory.seed = seed
    try :
        body_langs = langdetect.detect_langs(comment)
        # Three cases : 1)highest proba is french or english : we trust the classifier
        #               2)French or english proba are not zero but not the highest : we should look at it manually just in case  
        #               3) French proba or english proba is 0 : no doubt, this is another body_lang
        # Loop over each body_lang detected and compare probabilities 
        highest_prob = 0.
        prob_match_fr = 0.
        prob_match_en = 0.
        for l in body_langs:
            if l.prob > highest_prob : 
                highest_prob = l.prob
            if l.lang == 'fr' :
                prob_match_fr = l.prob
            if l.lang == 'en' :
                prob_match_en = l.prob
        if highest_prob == prob_match_fr :
            value='fr'   
        elif highest_prob == prob_match_en :
            value='en'
        elif prob_match_fr > 0. or prob_match_en > 0. :
            value='U' # undefined, might be matching body_lang but unsure
        else :
            value='N' # recognized another body_lang
    except : # if classification failed : due to emojis, images or whatever
        value='U' # undefined, have to look by myself
    return value


def human_class_df(df):
    ''' Ask the user to classify by hand a list of comments. Returns the dataframe that is passed as argument
        whith no undefined values anymore : everything is classed as 'fr','en' or 'N'.
    '''
    for index,row in df[df['body_lang']=='U'].iterrows():
         df.at[index,'body_lang']=human_classification(row['body'])
    return df

def human_classification(comment):
    ans = '-1'
    while(ans!='0' and ans!='1' and ans!='2'):
        print("Is the comment ",comment," written in English (1), French (2) or another body_lang (0) ?")
        ans = input()
    if ans == '2' :
        value='fr'
    if ans == '1' :
        value='en'
    if ans == '0' :
        value='N'
    return value