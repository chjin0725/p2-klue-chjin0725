import pickle as pickle
import os
import pandas as pd
import torch

class RE_Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset, labels):
        self.tokenized_dataset = tokenized_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.tokenized_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def insert_ENT(s, i1, i2, i3, i4):
    if i1 < i3:
        entity = ['[E1]', '[/E1]', '[E2]', '[/E2]']
    else:
        entity = ['[E2]', '[/E2]', '[E1]', '[/E1]']
    i1,i2,i3,i4 = sorted([i1,i2,i3,i4])
    s= s[:i1] + entity[0] + s[i1:i2+1] + entity[1] + s[i2+1:i3] + entity[2] + s[i3:i4+1] + entity[3] + s[i4+1:]
    return s

def preprocessing_dataset(dataset, label_type):
    label = []
    for i in dataset[8]:
        if i == 'blind':
            label.append(100)
        else:
            label.append(label_type[i])
    out_dataset = pd.DataFrame({'sentence':dataset[1],'entity_01':dataset[2],'entity_02':dataset[5],'label':label,})
    return out_dataset   

def load_data(dataset_dir, dev = True):

    with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
        label_type = pickle.load(f)

    dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
    dataset = preprocessing_dataset(dataset, label_type)

    if dev:
        train_dataset=dataset.sample(frac=0.8,random_state=42) #random state is a seed value
        dev_dataset=dataset.drop(train_dataset.index)
        return train_dataset, dev_dataset
    else:
        return dataset


def tokenized_dataset(dataset, tokenizer):
    concat_entity = []
    for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
        temp = ''
        temp = e01 + '[SEP]' + e02
        concat_entity.append(temp)
    tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=100,
      add_special_tokens=True,
      )
    return tokenized_sentences

def tokenized_dataset_ENT_token(dataset, tokenizer, model=None, train=True):
    added_token_num = tokenizer.add_special_tokens({'additional_special_tokens' :['[ENT]']})
    if train:
        model.resize_token_embeddings(tokenizer.vocab_size + added_token_num)
    
    concat_entity = []
    for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
        temp = ''
        temp = '[CLS]' + e01 + '[ENT]' + e02 + '[ENT]'
        concat_entity.append(temp)
    tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']+'[SEP]'),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=102,
      add_special_tokens=False,
      )
    
    return tokenized_sentences

def tokenized_dataset_ENT_token_in_sentence(dataset, tokenizer):
    

    tokenized_sentences = tokenizer(
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=112,
      add_special_tokens=True,
      )
    return tokenized_sentences