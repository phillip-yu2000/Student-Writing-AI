import os 
import gc 
import torch
import numpy as np 
import pandas as pd 

from ast import literal_eval
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, \
                         AutoConfig

from torch.utils.data import Dataset, DataLoader
from torch import cuda
from sklearn.metrics import accuracy_score

# RID WARNINGS IN DATASET CLASS WHEN USING TOKENIZER BEFORE FORKS
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DOWNLOADED_MODEL_PATH = '../longformer-base-4096'

VER = 1

LABEL_ALL_SUBTOKENS = True

config = {'model_name': 'allenai/longformer-base-4096',   
         'max_length': 1024,
         'train_batch_size':4,
         'valid_batch_size':4,
         'epochs':1,
         'learning_rates': [2.5e-5, 2.5e-5, 2.5e-6, 2.5e-6, 2.5e-7],
         'max_grad_norm':10,
         'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")}

# CREATE DICTIONARIES THAT WE CAN USE DURING TRAIN AND INFER
output_labels = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 
                 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim', 
                 'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 
                 'B-Concluding Statement', 'I-Concluding Statement']

labels_to_ids = {v:k for k,v in enumerate(output_labels)}
ids_to_labels = {k:v for k,v in enumerate(output_labels)}

# LOAD DATA GENERATED FROM SETUP-CODE
train_df = pd.read_csv('../feedback-prize-2021/train.csv')

train_fold = pd.read_csv('../preprocessed/train_folds.csv')
train_fold = train_fold[['id', 'kfold']] 

test_texts = pd.read_csv('../preprocessed/test_text.csv')
train_texts = pd.read_csv('../preprocessed/train_NER.csv')

# PANDAS STORES AS STRING NEED TO CONVERT IT TO OBJECT DTYPE
train_texts.entities = train_texts.entities.apply(lambda x: literal_eval(x))

# CREATE LIST OF TEXTS SEPARATED BY FOLD NUMBER
folds_texts = list()

for fold_num in range(len(train_fold.kfold.unique())):
    fold = train_fold[train_fold.kfold == fold_num]
    fold_texts = train_texts.loc[train_texts['id'].isin(fold['id'])]
    folds_texts.append(fold_texts)

class FeedbackDataset(Dataset):
    '''
    PyTorch Dataset Class
    '''
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.len = len(dataframe)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        # GETS TEXT AND WORD LABELS
        text = self.data.text.iloc[index]
        word_labels = self.data.entities.iloc[index]
        
        # TOKENIZE TEXT
        encoding = self.tokenizer(text.split(),
                                 is_split_into_words = True,
                                 padding = 'max_length',
                                 truncation = True,
                                 max_length = self.max_len
                                 )
        word_ids = encoding.word_ids()
        
        # CREATE LABELS
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(labels_to_ids[word_labels[word_idx]])
            else:
                if LABEL_ALL_SUBTOKENS:
                    label_ids.append(labels_to_ids[word_labels[word_idx]])
                else:
                    label_ids.append(-100)
            
            previous_word_idx = word_idx
        encoding['labels'] = label_ids
        
        # CONVERT TO TORCH TENSORS
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        word_ids = [w if w is not None else -1 for w in word_ids]
        item['wids'] = torch.as_tensor(word_ids)
        
        return item
    
    def __len__(self):
        return self.len

# TRAIN AND VALID PARAMETERS
train_params = {'batch_size': config['train_batch_size'],
                'shuffle': True,
                'num_workers': 2,
                'pin_memory':True
                }

valid_params = {'batch_size': config['valid_batch_size'],
                'shuffle': False,
                'num_workers': 2,
                'pin_memory':True
                }

tokenizer = AutoTokenizer.from_pretrained(DOWNLOADED_MODEL_PATH)

def train(epoch):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    
    # MODEL IN TRAINING MODE
    model.train()

    for idx, batch in enumerate(training_loader):

        ids = batch['input_ids'].to(config['device'], dtype = torch.long)
        mask = batch['attention_mask'].to(config['device'], dtype = torch.int64)
        labels = batch['labels'].to(config['device'], dtype = torch.long)
        
        
        loss, tr_logits = model(input_ids = ids, attention_mask = mask, 
                                labels = labels, return_dict = False)

        tr_loss += loss.item()
        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)

        # PROGRESS TRACKER PRINTED TO OUTPUT
        if idx % 200 == 0:
            loss_step = tr_loss/nb_tr_steps
            print(f"Training loss after {idx:04d} training steps: {loss_step}")

        # COMPUTE TRAINING ACCURACY
        flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
        active_logits = tr_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
        active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)

        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy

        # GRADIENT CLIPPING
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=config['max_grad_norm']
        )

        # BACKWARDS PASS
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    print(f"Training loss epoch: {epoch_loss}")
    print(f"Training accuracy epoch: {tr_accuracy}")

def inference(batch):
                
    # MOVE BATCH TO GPU AND INFER
    ids = batch["input_ids"].to(config['device'])
    mask = batch["attention_mask"].to(config['device'])
    outputs = model(ids, attention_mask=mask, return_dict=False)
    all_preds = torch.argmax(outputs[0], axis=-1).cpu().numpy() 

    # INTERATE THROUGH EACH TEXT AND GET PRED
    predictions = []
    for k,text_preds in enumerate(all_preds):
        token_preds = [ids_to_labels[i] for i in text_preds]

        prediction = []
        word_ids = batch['wids'][k].numpy()  
        previous_word_idx = -1
        for idx,word_idx in enumerate(word_ids):                            
            if word_idx == -1:
                pass
            elif word_idx != previous_word_idx:              
                prediction.append(token_preds[idx])
                previous_word_idx = word_idx
        predictions.append(prediction)
    
    return predictions

def get_predictions(df, loader):
    
    # put model in training mode
    model.eval()
    
    # GET WORD LABEL PREDICTIONS
    y_pred2 = []
    for batch in loader:
        labels = inference(batch)
        y_pred2.extend(labels)

    final_preds2 = []
    for i in range(len(df)):

        idx = df.id.values[i]
        pred = y_pred2[i] # Leave "B" and "I"
        preds = []
        j = 0
        while j < len(pred):
            cls = pred[j]
            if cls == 'O': j += 1
            else: cls = cls.replace('B','I') # spans start with B
            end = j + 1
            while end < len(pred) and pred[end] == cls:
                end += 1
            
            if cls != 'O' and cls != '' and end - j > 7:
                final_preds2.append((idx, cls.replace('I-',''),
                                     ' '.join(map(str, list(range(j, end))))))
        
            j = end
        
    oof = pd.DataFrame(final_preds2)
    oof.columns = ['id','class','predictionstring']

    return oof

def calc_overlap(row):
    """
    Calculates the overlap between prediction and
    ground truth and overlap percentages used for determining
    true positives.
    """
    set_pred = set(row.predictionstring_pred.split(' '))
    set_gt = set(row.predictionstring_gt.split(' '))
    # Length of each and intersection
    len_gt = len(set_gt)
    len_pred = len(set_pred)
    inter = len(set_gt.intersection(set_pred))
    overlap_1 = inter / len_gt
    overlap_2 = inter/ len_pred
    return [overlap_1, overlap_2]


def score_feedback_comp(pred_df, gt_df):
    """
    A function that scores for the kaggle
        Student Writing Competition
        
    Uses the steps in the evaluation page here:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    """
    gt_df = gt_df[['id','discourse_type','predictionstring']] \
        .reset_index(drop=True).copy()
    pred_df = pred_df[['id','class','predictionstring']] \
        .reset_index(drop=True).copy()
    pred_df['pred_id'] = pred_df.index
    gt_df['gt_id'] = gt_df.index
    # Step 1. all ground truths and predictions for a given class are compared.
    joined = pred_df.merge(gt_df,
                           left_on=['id','class'],
                           right_on=['id','discourse_type'],
                           how='outer',
                           suffixes=('_pred','_gt')
                          )
    joined['predictionstring_gt'] = joined['predictionstring_gt'].fillna(' ')
    joined['predictionstring_pred'] = joined['predictionstring_pred'].fillna(' ')

    joined['overlaps'] = joined.apply(calc_overlap, axis=1)

    # 2. If the overlap between the ground truth and prediction is >= 0.5, 
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    joined['overlap1'] = joined['overlaps'].apply(lambda x: eval(str(x))[0])
    joined['overlap2'] = joined['overlaps'].apply(lambda x: eval(str(x))[1])


    joined['potential_TP'] = (joined['overlap1'] >= 0.5) & (joined['overlap2'] >= 0.5)
    joined['max_overlap'] = joined[['overlap1','overlap2']].max(axis=1)
    tp_pred_ids = joined.query('potential_TP') \
        .sort_values('max_overlap', ascending=False) \
        .groupby(['id','predictionstring_gt']).first()['pred_id'].values

    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    fp_pred_ids = [p for p in joined['pred_id'].unique() if p not in tp_pred_ids]

    matched_gt_ids = joined.query('potential_TP')['gt_id'].unique()
    unmatched_gt_ids = [c for c in joined['gt_id'].unique() if c not in matched_gt_ids]

    # Get numbers of each type
    TP = len(tp_pred_ids)
    FP = len(fp_pred_ids)
    FN = len(unmatched_gt_ids)
    #calc microf1
    my_f1_score = TP / (TP + 0.5*(FP+FN))
    return my_f1_score

# CREATE TRAIN AND VALID DATASET
for nb_fold in range(len(folds_texts)):
    print(f"### Training Fold: {nb_fold + 1} out of {len(folds_texts)} ###")
    
    # CREATE MODEL FROM DOWNLOADED PATH
    config_model = AutoConfig.from_pretrained(DOWNLOADED_MODEL_PATH+'/config.json')
    model = AutoModelForTokenClassification.from_pretrained(
        DOWNLOADED_MODEL_PATH+'/pytorch_model.bin', config = config_model
    )
    model.to(config['device'])
    optimizer = torch.optim.Adam(
        params = model.parameters(), lr = config['learning_rates'][0]
    )

    # FOLD WILL BE VALIDATION THE REST ARE TRAINING
    validation_set = FeedbackDataset(
        folds_texts[nb_fold], tokenizer, config['max_length']
    )
    training_set = FeedbackDataset(
        pd.concat(folds_texts[:nb_fold] + folds_texts[nb_fold+1:]),
        tokenizer, config['max_length']
    )

    # BATCH SAMPLES USING PYTORCH DATALOADER
    validation_loader = DataLoader(validation_set, **valid_params)
    training_loader = DataLoader(training_set, **train_params)
    
    for epoch in range(config['epochs']):
        print(f"## Training epoch: {epoch + 1}")
        
        for g in optimizer.param_groups:
            g['lr'] = config['learning_rates'][epoch]
        
        lr = optimizer.param_groups[0]['lr']
        print(f'### LR = {lr}\n')
        
        train(epoch)
        torch.cuda.empty_cache()
        gc.collect()
    
    torch.save(model.state_dict(), f'longformer{nb_fold}_v{VER}.pt')
    
    # VALIDATION TARGETS
    valid = train_df.loc[train_df['id'].isin(folds_texts[nb_fold].id)]
    
    # Out-Of-Fold (OOF) PREDICTIONS
    oof = get_predictions(folds_texts[nb_fold], validation_loader)
    
    # COMPUTE F1 SCORE
    f1s = []
    CLASSES = oof['class'].unique()
    print()
    for c in CLASSES:
        pred_df = oof.loc[oof['class']==c].copy()
        gt_df = valid.loc[valid['discourse_type']==c].copy()
        f1 = score_feedback_comp(pred_df, gt_df)
        print(c,f1)
        f1s.append(f1)
    print()
    print(f'Validation Fold {nb_fold + 1}:',np.mean(f1s))
    print()