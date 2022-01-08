import bz2
import os.path
import numpy as np
import re
from torchtext.data import get_tokenizer
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from pathlib import Path
import psutil

data_path = os.path.dirname(os.path.abspath(__file__))
print()

#from torchnlp.encoders.text import StaticTokenizerEncoder, stack_and_pad_tensors, pad_tensor


##############################
#    read raw gros bizou2    #
##############################
def get_packed(xs: [torch.Tensor]) -> "PackedSequence":
    padded = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
    packd = torch.nn.utils.rnn.pack_padded_sequence(
        padded, batch_first=True, lengths=list(map(len, xs)), enforce_sorted=False)
    return packd

def collate(batch):
    batch = [torch.tensor(x, dtype=torch.float32).view(-1,1) for x in batch]
    return get_packed(batch)

def get_labels_and_texts(file):
    labels = []
    texts = []
    k,j = 0,0 #slice else it takes too long to run
    if psutil.virtual_memory().available < 8000000:
        '''adding a ram checker for paul's smol computer because I want more of the rows'''
        k = 6990
    else:
        k= 100000

    for line in bz2.BZ2File(file):
        if j == k: #totally arbitrary stop point because my computer is smol bean :(
                break 
        x = line.decode("utf-8")
        labels.append(int(x[9]) - 1)
        texts.append(x[10:].strip())
        j += 1
    return np.array(labels), texts


def normalize_texts(texts):
    NON_ALPHANUM = re.compile(r'[\W]')
    NON_ASCII = re.compile(r'[^a-z0-1\s]')
    normalized_texts = []
    for text in texts:
        lower = text.lower()
        no_punctuation = NON_ALPHANUM.sub(r' ', lower)
        no_non_ascii = NON_ASCII.sub(r'', no_punctuation)
        normalized_texts.append(no_non_ascii)

    return normalized_texts


def amzreview_dataset() -> (DataLoader, DataLoader):
    """
    reads files from /data/raw with appropriate method 
    applies normalization and tokenization
    merges processed texts with their labels into 
    torch.utils.data DataLoader object for train and test sets
    """

    train_labels, train_texts = get_labels_and_texts(data_path + '/raw/train.ft.txt.bz2')
    test_labels, test_texts = get_labels_and_texts(data_path + '/raw/test.ft.txt.bz2')

    #######################
    #    preprocessing    # should be part of dataloader tbqh
    #######################
            
    train_texts = normalize_texts(train_texts)
    test_texts = normalize_texts(test_texts)

    
    ##########################
    #    train/test split    # - for debugging; now done in dataloader senere
    ##########################
    #from sklearn.model_selection import train_test_split
    #train_texts, val_texts, train_labels, val_labels = train_test_split(
    #    train_texts, train_labels, random_state=57643892, test_size=0.2)

    ##################
    #    tokenize    # (already normalized by hand above)
    ##################
    Tokenizer = get_tokenizer(None) #maybe look into tokenizination libraries (spacy, etc) 
    #following prints are just for debugging
    print("len before tok:", len(train_texts))
    train_data= [ [Tokenizer(text), train_labels[i]] for i,text in enumerate(train_texts)]
    test_data = [ [Tokenizer(text), test_labels[i]] for i,text in enumerate(test_texts)]
    print("len after  tok:", len(train_data),"\n")
    ###################
    #   dataloader    #
    ###################

    train = DataLoader(train_data, shuffle=True, batch_size=3000, collate_fn=collate)
    test = DataLoader(test_data, shuffle=True, batch_size=3000, collate_fn=collate)
    return train, test

amzreview_dataset()


