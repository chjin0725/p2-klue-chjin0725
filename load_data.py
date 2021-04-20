import pickle as pickle
import os
import pandas as pd
import torch

# Dataset 구성.
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

# 처음 불러온 tsv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
# 변경한 DataFrame 형태는 baseline code description 이미지를 참고해주세요.

def insert_ENT(s, i1, i2, i3, i4):
    if i1 < i3:
        entity = ['[E1]', '[/E1]', '[E2]', '[/E2]']
    else:
        entity = ['[E2]', '[/E2]', '[E1]', '[/E1]']
    # entity = ['[ENT]', '[/ENT]', '[ENT]', '[/ENT]']
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
    # dataset[1] = dataset.apply(lambda x: insert_ENT(x[1], x[3], x[4], x[6], x[7]), axis = 1)
    # new_data = pd.read_csv("/opt/ml/input/data/train/19_26_37_40.csv")
    out_dataset = pd.DataFrame({'sentence':dataset[1],'entity_01':dataset[2],'entity_02':dataset[5],'label':label,})
    return out_dataset    #pd.concat([new_data,out_dataset]).reset_index().drop(['index'], axis=1)

# tsv 파일을 불러옵니다.
def load_data(dataset_dir, dev = True):
    # load label_type, classes
    with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
        label_type = pickle.load(f)
    # load dataset
    dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
    # preprecessing dataset
    dataset = preprocessing_dataset(dataset, label_type)

    if dev:
        train_dataset=dataset.sample(frac=0.8,random_state=42) #random state is a seed value
        dev_dataset=dataset.drop(train_dataset.index)
        return train_dataset, dev_dataset
    else:
        return dataset

# bert input을 위한 tokenizing.
# tip! 다양한 종류의 tokenizer와 special token들을 활용하는 것으로도 새로운 시도를 해볼 수 있습니다.
# baseline code에서는 2가지 부분을 활용했습니다.
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