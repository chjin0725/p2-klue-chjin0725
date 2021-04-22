import torch
from torch import nn
from transformers import BertModel, BertConfig, BertForSequenceClassification
from transformers.models.bert.modeling_bert import BertPreTrainedModel


class MaskedMeanPooling(nn.Module):
    def __init__(self):
        super().__init__()
        
    
    def forward(self, x, mask):
        num_not_padding = torch.sum(mask, dim=1).view(-1,1)
        mask = mask.view(mask.size(0), mask.size(1),-1)
        
        x_masked = x * mask
        sum_x_masked = torch.sum(x_masked, dim = 1)
        out = torch.true_divide(sum_x_masked, num_not_padding)
        
        return out

class MaskedMeanPoolingTokenTypeID(nn.Module):
    ''' [CLS]e1[SEP]e2[SEP] 부분만 사용해서 평균.
    '''
    def __init__(self):
        super().__init__()
        
    
    def forward(self, x, attention_mask, token_type_ids):
        '''여기서 mask는 token type id이다.'''
        token_type_ids = (token_type_ids != attention_mask).int()
        num_to_sep = torch.sum(token_type_ids, dim=1).view(-1,1) ## [CLS]e1[SEP]e2[SEP] 부분의 토큰 개수.
        token_type_ids = token_type_ids.view(token_type_ids.size(0), token_type_ids.size(1),-1)
        
        x_masked = x * token_type_ids
        sum_x_masked = torch.sum(x_masked, dim = 1)
        out = torch.true_divide(sum_x_masked, num_to_sep)
        
        return out

class ClsSepConcat(nn.Module):
    '''CLS토큰과 sentence 시작부분 바로 앞의 SEP토큰을 concat
    '''
    def __init__(self):
        super().__init__()
        
    def forward(self, x, attention_mask, token_type_ids):
        SEP_index_for_each_batch = (token_type_ids != attention_mask).sum(dim = -1) - 1
        SEP_index_for_each_batch = SEP_index_for_each_batch.view(-1,1)
        
        batch_index = torch.LongTensor(range(token_type_ids.size(0))).unsqueeze(1)
        
        idx = [batch_index, SEP_index_for_each_batch]
        
        CLS = x[:,0,:]
        SEP = x[idx].squeeze(1)
        
        return torch.cat((CLS, SEP), dim = -1)

class ClsToNthConcat(nn.Module):
    '''Concat hidden states from CLS token to Nth token.
    '''
    def __init__(self):
        super().__init__()
        
    def forward(self, x, N):
        
        hidden_states = []
        for i in range(N):
            hidden_states.append(x[:,i,:])
        
        return torch.cat(hidden_states, dim = -1)
    
class MyBertMean(nn.Module):

    def __init__(self, model_name, bert_config):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(model_name)
        self.masked_mean_pooling = MaskedMeanPooling()
        self.classifier = nn.Linear(bert_config.hidden_size, bert_config.num_labels)
        
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        '''
        Args:
            input_ids (Tensor): shape (batch_size, seq_len).
            attention_mask (Tensor): shape (batch_size, seq_len).
            token_type_ids (Tensor): shape (batch_size, seq_len).
        
        Returns:
            logit (Tensor): shape (batch_size, num_labels).
                
        '''
        x = self.bert(input_ids=input_ids,
                      attention_mask=attention_mask,
                      token_type_ids=token_type_ids,)[0] ## shape (batch_size, seq_len, emb_dim).  pooled output이 아닌 last_hidden_state를 가져온다.
        x = self.masked_mean_pooling(x, attention_mask) ## shape(batch_size, emb_dim).
        logit = self.classifier(x)
        
        return logit
        
        
class BertForSC(nn.Module): ## BertForSequencClassification 모델과 똑같은 구조.

    def __init__(self, model_name, bert_config):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        self.classifier = nn.Linear(bert_config.hidden_size, bert_config.num_labels)
        
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        '''
        Args:
            input_ids (Tensor): shape (batch_size, seq_len).
            attention_mask (Tensor): shape (batch_size, seq_len).
            token_type_ids (Tensor): shape (batch_size, seq_len).
        
        Returns:
            logit (Tensor): shape (batch_size, num_labels).
                
        '''
        x = self.bert(input_ids=input_ids,
                      attention_mask=attention_mask,
                      token_type_ids=token_type_ids,)[1] ## shape (batch_size, emb_dim).
        x = self.dropout(x)
        logit = self.classifier(x)
        
        return logit
    
class MyBertMeanTokenTypeID(nn.Module):

    def __init__(self, model_name, bert_config):
        super().__init__()
    
        self.bert = BertModel.from_pretrained(model_name)
        self.masked_mean_pooling_token_type_id = MaskedMeanPoolingTokenTypeID()
        self.classifier = nn.Linear(bert_config.hidden_size, bert_config.num_labels)
        
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        '''
        Args:
            input_ids (Tensor): shape (batch_size, seq_len).
            attention_mask (Tensor): shape (batch_size, seq_len).
            token_type_ids (Tensor): shape (batch_size, seq_len).
        
        Returns:
            logit (Tensor): shape (batch_size, num_labels).
                
        '''
        x = self.bert(input_ids=input_ids,
                      attention_mask=attention_mask,
                      token_type_ids=token_type_ids,)[0] ## shape (batch_size, seq_len, emb_dim).  pooled output이 아닌 last_hidden_state를 가져온다.
        x = self.masked_mean_pooling_token_type_id(x, attention_mask, token_type_ids) ## shape(batch_size, emb_dim).
        logit = self.classifier(x)
        
        return logit
    
class MyBertClsSepConcat(nn.Module):

    def __init__(self, model_name, bert_config):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(model_name)
        self.CLS_SEP_concat = ClsSepConcat()
        self.classifier = nn.Linear(bert_config.hidden_size*2, bert_config.num_labels)
        
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        '''
        Args:
            input_ids (Tensor): shape (batch_size, seq_len).
            attention_mask (Tensor): shape (batch_size, seq_len).
            token_type_ids (Tensor): shape (batch_size, seq_len).
        
        Returns:
            logit (Tensor): shape (batch_size, num_labels).
                
        '''
        x = self.bert(input_ids=input_ids,
                      attention_mask=attention_mask,
                      token_type_ids=token_type_ids,)[0] ## shape (batch_size, seq_len, emb_dim).  pooled output이 아닌 last_hidden_state를 가져온다.
        x = self.CLS_SEP_concat(x, attention_mask, token_type_ids) ## shape(batch_size, emb_dim).
        logit = self.classifier(x)
        
        return logit

class MyBertClsToNthConcat(nn.Module):

    def __init__(self, model_name, bert_config, N):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(model_name)
        self.CLS_to_Nth_concat = ClsToNthConcat()
        self.classifier = nn.Linear(bert_config.hidden_size*N, bert_config.num_labels)
        self.N = N
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        '''
        Args:
            input_ids (Tensor): shape (batch_size, seq_len).
            attention_mask (Tensor): shape (batch_size, seq_len).
            token_type_ids (Tensor): shape (batch_size, seq_len).
        
        Returns:
            logit (Tensor): shape (batch_size, num_labels).
                
        '''
        x = self.bert(input_ids=input_ids,
                      attention_mask=attention_mask,
                      token_type_ids=token_type_ids,)[0]  ## shape (batch_size, seq_len, emb_dim).  pooled output이 아닌 last_hidden_state를 가져온다.
        x = self.CLS_to_Nth_concat(x, self.N)  ## shape(batch_size, emb_dim).
        logit = self.classifier(x)
        
        return logit