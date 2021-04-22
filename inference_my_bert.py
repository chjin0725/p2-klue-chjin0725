import pickle as pickle
import copy
import argparse
import random
import os

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer, BertModel, BertConfig

from my_bert import *
from load_data import *


def inference(model, tokenized_sent, device):
    dataloader = DataLoader(tokenized_sent, batch_size=40, shuffle=False)
    model.eval()
    output_pred = []
  
    for i, data in enumerate(dataloader):
        with torch.no_grad():
            outputs = model(
              input_ids=data['input_ids'].to(device),
              attention_mask=data['attention_mask'].to(device),
              token_type_ids=data['token_type_ids'].to(device)
              )
        logits = outputs ## shape (batch_size, num_labels)
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
  
    return np.array(output_pred).flatten()



def load_test_dataset(dataset_dir, tokenizer):
    test_dataset = load_data(dataset_dir, dev= False)
    test_label = test_dataset['label'].values

    # tokenizing dataset
    tokenized_test = tokenized_dataset(test_dataset, tokenizer)

    print(tokenizer.convert_ids_to_tokens(tokenized_test['input_ids'][2]))


    return tokenized_test, test_label


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    test_dataset_dir = "/opt/ml/input/data/test/test.tsv"
    test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
    test_dataset = RE_Dataset(test_dataset ,test_label)
    
    bert_config = BertConfig.from_pretrained(args.model_name)
    bert_config.num_labels = 42


    model = torch.load(args.model_dir + args.model_name + args.version + '.pt')
    temp = model.to(device)
    
    pred_answer = inference(model, test_dataset, device)
    output = pd.DataFrame(pred_answer, columns=['pred'])

    output.to_csv('./prediction/' + args.model_name + args.version + '.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', type=str, default="./results/")
    parser.add_argument('--model_name', type=str, default="bert-base-multilingual-cased")
    parser.add_argument('--version', type=str, default="_v54")
    args = parser.parse_args()
    main(args)