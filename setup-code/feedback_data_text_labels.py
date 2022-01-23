import enum
from html import entities
import os 

import numpy as np 
import pandas as pd 

from tqdm import tqdm 

def main():
    '''
    Create dataframes with essay ids and text.
    Training data includes NER labels.
    Save them to a directory called preprocessed.
    '''
    
    # CREATE DIRECTORY IF NOT EXISTS
    if not os.path.exists('../preprocessed'):
        os.mkdir('../preprocessed')

    # IMPORT DATA
    train_df = pd.read_csv('../feedback-prize-2021/train.csv')

    # CREATE DATAFRAME WITH TEST ID AND TEXT
    test_names, test_texts = [], []
    for f in list(os.listdir('../feedback-prize-2021/test')):
        test_names.append(f.replace('.txt', ''))
        test_texts.append(open('../feedback-prize-2021/test/' + f, 'r').read())
    test_texts = pd.DataFrame({'id': test_names, 'text':test_texts})
    test_texts.to_csv('../preprocessed/test_texts.csv', index=False)

    # CREATE DATAFRAME WITH TRAINING ID AND TEXT
    train_names, train_texts = [], []
    print('Starting to process training ids and text.')
    for f in tqdm(list(os.listdir('../feedback-prize-2021/train'))):
        train_names.append(f.replace('.txt', ''))
        train_texts.append(open('../feedback-prize-2021/train/' + f, 'r').read())
    train_texts = pd.DataFrame({'id': train_names, 'text': train_texts})
    print('Training ids and text finished processing.', '\n')


    # CREATE NAMED ENTITY LABELS FOR ESSAYS
    all_entities = []
    print('Starting to process training labels.')
    for ii,i in enumerate(train_texts.iterrows()):
        
        if ii%100==0: print(ii,', ',end='') # PROGRESS TRACKER
        
        total = i[1]['text'].split().__len__()
        entities = ["O"]*total
        
        for j in train_df[train_df['id'] == i[1]['id']].iterrows():
            discourse = j[1]['discourse_type']
            list_ix = [int(x) for x in j[1]['predictionstring'].split(' ')]
            entities[list_ix[0]] = f"B-{discourse}"
            for k in list_ix[1:]: 
                try:
                    entities[k] = f"I-{discourse}"
                except IndexError:
                    entities.append(f"I-{discourse}")
        
        all_entities.append(entities)
    
    train_texts['entities'] = all_entities
    train_texts.to_csv(r'../preprocessed/train_NER.csv',index=False)
    print('Training labels finished processing and saved.')


if __name__ == '__main__':
    main()
