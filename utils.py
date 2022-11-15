import logging
logging.disable(logging.INFO) 
logging.disable(logging.WARNING)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import torch.utils.data

def get_maxlen_classes(task):
    if(task in ['ones_zeros', 'unique', 'frequent']):
      max_sen_len = 35
    elif(task in ['anagram', 'palindrome', 'digits', 'vowels', 'length', 'arthwords', 'arthnumwords', '012Star02Star']):
      max_sen_len = 15
    elif(task in ['tautonym', 'mean']):
      max_sen_len = 20
    elif(task in ['isogram']):
      max_sen_len = 50
    elif(task in ['mode', 'median']):
      max_sen_len = 40
    elif(task in ["AAstarBBstarCCstarDDstarEEstar"]):
      max_sen_len = 30
    elif(task in ['arthnum', 'odd_even']):
      max_sen_len = 5

    if(task in ['length', 'unique', 'frequent']):
      numclasses = 10
    elif(task in ['anagram', 'palindrome', 'digits', 'vowels', 'ones_zeros', 'isogram', 'tautonym', 'odd_even']):
      numclasses = 2
    elif(task in ["AAstarBBstarCCstarDDstarEEstar", '012Star02Star']):
      numclasses = 2
    elif(task in ['arthnum','arthwords', 'arthnumwords', 'mode', 'median', 'mean']):
      numclasses = 10

    return max_sen_len, numclasses

def load_portion(data, labels, portion_size):
    data1, data2, labels1, labels2 = train_test_split(data, labels, test_size=portion_size, random_state=42)
    return data2, labels2

def tokenize_sentences(tokenizer, sentences, maxlen):
    input_ids = []
    attention_masks = []

    for sentence in sentences:
      sentence = str(sentence)
      encoded = tokenizer.encode_plus(
        text=sentence,
        add_special_tokens=True,
        padding='max_length',
        max_length=maxlen,
        truncation=True,
        return_token_type_ids=True,
        return_tensors='pt'
      )

      input_ids.append(encoded['input_ids'])
      attention_masks.append(encoded['attention_mask'])

      input_ids = torch.cat(input_ids, dim=0, out=None)
      attention_masks = torch.cat(attention_masks, dim=0, out=None)

    return input_ids, attention_masks

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    avg = 'macro'
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=avg)
    acc = accuracy_score(labels, preds) * 100
    return {
        'accuracy': acc,
        'f1': f1 * 100,
        'precision': precision,
        'recall': recall
    }

class TaskDataset(torch.utils.data.Dataset):
    def __init__(self, inputids, attnmasks, labels):
      self.inputids = inputids
      self.attnmasks = attnmasks
      self.labels = labels

    def __getitem__(self, idx):
      item = {}
      item['input_ids'] = self.inputids[idx]
      item['attention_mask'] = self.attnmasks[idx]
      item['labels'] = self.labels[idx]
      return item

    def __len__(self):
      return len(self.labels)
