import os 
import torch
from transformers import AutoTokenizer, AutoConfig, \
    AutoModelForTokenClassification

def main():
    MODEL_NAME = 'allenai/longformer-base-4096'

    # DOWNLOAD TOKENIZER, CONFIGURATION, AND MODEL
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, add_prefix_space = True
    )
    
    config_model = AutoConfig.from_pretrained(
        MODEL_NAME, num_labels = 15
    )

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels = 15
    )

    # SAVE TOKENIZER, CONFIGURATION, AND MODEL
    os.mkdir('../longformer')
    
    tokenizer.save_pretrained('../longformer')
    
    config_model.save_pretrained('../longformer')
    
    model.save_pretrained('../longformer')

if __name__ == '__main__':
    main()