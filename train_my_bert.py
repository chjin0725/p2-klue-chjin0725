import pickle as pickle
import copy
import argparse
import random
import os
os.environ['WANDB_PROJECT'] = 'Pstage2_KLUE'
# os.environ['WANDB_LOG_MODEL'] = 'true'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from transformers import AutoTokenizer, BertModel, BertConfig

from my_bert import *
from avgMeter import AverageMeter
from load_data import *
import wandb

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def train(args, model, criterion, optimizer, scheduler, train_loader, valid_loader):
    
    train_loss, train_acc = AverageMeter(), AverageMeter()
    valid_loss, valid_acc = AverageMeter(), AverageMeter()
    best_val_acc = 0
    for epoch in range(args.num_epochs):
        train_loss.reset()
        train_acc.reset()
        for iter, x in enumerate(train_loader):

            input_ids, attention_mask, token_type_ids = x['input_ids'].cuda(), x['attention_mask'].cuda(), x['token_type_ids'].cuda()
            label = x['labels'].cuda()

            pred_logit = model(input_ids, attention_mask, token_type_ids)

            loss = criterion(pred_logit, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + iter/len(train_loader))

            pred_label = pred_logit.argmax(-1)
            acc = (pred_label == label).sum().float() / input_ids.size(0)

            train_loss.update(loss.item(), input_ids.size(0))
            train_acc.update(acc, input_ids.size(0))



        valid_loss.reset()
        valid_acc.reset()
        for x in valid_loader:
            input_ids, attention_mask, token_type_ids = x['input_ids'].cuda(), x['attention_mask'].cuda(), x['token_type_ids'].cuda()
            label = x['labels'].cuda()

            with torch.no_grad():
                pred_logit = model(input_ids, attention_mask, token_type_ids)


            loss = criterion(pred_logit, label)

            pred_label = pred_logit.argmax(-1)
            acc = (pred_label == label).sum().float() / input_ids.size(0)

            valid_loss.update(loss.item(), input_ids.size(0))
            valid_acc.update(acc, input_ids.size(0))


        train_loss_val = train_loss.avg
        train_acc_val = train_acc.avg
        valid_loss_val = valid_loss.avg
        valid_acc_val = valid_acc.avg
        
        wandb.log(
                {
                    "Train ACC": train_acc_val,
                    "Valid ACC": valid_acc_val,
                    "Train Loss": train_loss_val,
                    "Valid Loss": valid_loss_val,
                }
            )
        
        print("epoch [%3d/%3d] | Train Loss %.4f | Train Acc %.4f | Valid Loss %.4f | Valid Acc %.4f" %
            (epoch+1, args.num_epochs, train_loss_val, train_acc_val, valid_loss_val, valid_acc_val))


        if valid_acc_val > best_val_acc:
            best_val_acc = valid_acc_val
            best_model_wts = copy.deepcopy(model.state_dict())
            
    model.load_state_dict(best_model_wts)    
    torch.save(model, './results/' + args.model_name + args.version + '.pt')


def data_loader(tokenizer, args):
    train_dataset, dev_dataset = load_data("/opt/ml/input/data/train/train.tsv")
    
    train_label = train_dataset['label'].values
    dev_label = dev_dataset['label'].values
    
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)
    
    print(tokenizer.convert_ids_to_tokens(tokenized_train['input_ids'][2]))
    print(tokenizer.convert_ids_to_tokens(tokenized_dev['input_ids'][2]))
    
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)
    
    train_loader = DataLoader(RE_train_dataset, batch_size=args.batch_size, num_workers = 4,  pin_memory=True, shuffle=True)
    valid_loader = DataLoader(RE_dev_dataset, batch_size=args.batch_size, num_workers = 4,  pin_memory=True, shuffle=False)
    
    return train_loader, valid_loader
    
def main(args):
    seed_everything(42)
    
    bert_config = BertConfig.from_pretrained(args.model_name)
    bert_config.num_labels = 42
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    train_loader, valid_loader = data_loader(tokenizer,args)
    
    model = MyBertClsSepConcat(args.model_name, bert_config)
    tmep = model.cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, args.num_epochs, 1)
    
    train(args, model, criterion, optimizer, scheduler, train_loader, valid_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model dir
    
    parser.add_argument('--model_name', type=str, default="bert-base-multilingual-cased")
    parser.add_argument('--version', type=str, default="_v26")
    parser.add_argument('--num_epochs', type=int, default = 10)
    parser.add_argument('--lr', type=float, default = 0.000025)
    parser.add_argument('--batch_size', type=int, default = 16)
    args = parser.parse_args()
    wandb.init(tags=["epoch 10", "[CLS]e1[SEP]e2[SEP]sentence[SEP]", "dev set", 'seed 42', 'lr 25e-6', 'CLS SEP concat', 'batch size 32'], name = args.model_name+args.version)
    main(args)