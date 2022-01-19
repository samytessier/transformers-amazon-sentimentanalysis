<<<<<<< HEAD
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


=======
import bz2
import os
import os.path
import numpy as np
import re

import torchtext
from torchtext.data import get_tokenizer
from torchtext.legacy.data import Field
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torchtext.vocab import FastText, vocab
import torch
from transformers import AutoTokenizer


#from torchtext.vocab import GloVe

##########################
#    read raw  bizou2    #
##########################


#print("when enetering chungus we're in: \n",os.getcwd())

def get_labels_and_texts(file) -> "np.array":
    """
    from .bz2 archive to np.array
    """
    labels = []
    texts = []
    k = 0 #slice else it takes too long to run
    for line in bz2.BZ2File(file):
        if k == 6990: #totally arbitrary stop point because my computer is smol bean :(
            break 
        x = line.decode("utf-8")
        labels.append(int(x[9]) - 1)
        texts.append(x[10:].strip())
        k += 1
    return np.array(labels), texts


def normalize_texts(texts):
    """
    returns array of normalized texts from array
    of dirty ugly ones
    """
    NON_ALPHANUM = re.compile(r'[\W]')
    NON_ASCII = re.compile(r'[^a-z0-1\s]')
    normalized_texts = []
    for text in texts:
        lower = text.lower()
        no_punctuation = NON_ALPHANUM.sub(r' ', lower)
        no_non_ascii = NON_ASCII.sub(r'', no_punctuation)
        normalized_texts.append(no_non_ascii)
    return normalized_texts


def amzreview_dataset(fpath) -> (DataLoader, DataLoader):
    """
    reads files from /data/raw with appropriate method 
    applies normalization and tokenization
    merges processed texts with their labels into 
    torch.utils.data DataLoader object for train and test sets
    """
    train_labels, train_texts = get_labels_and_texts(fpath)#'data/raw/test.ft.txt.bz2')
    test_labels, test_texts = get_labels_and_texts(fpath)#'data/raw/test.ft.txt.bz2')

    train_texts = normalize_texts(train_texts)
    test_texts = normalize_texts(test_texts)
    #print("#txts mod 64: ", len(train_texts) % 64)
    #MAX_SEQ_SIZE = max([len(t) for t in train_texts+test_texts])


    ####################
    #    tokenize      # (already normalized by hand above)
    ####################
    Tokenizer = get_tokenizer(None) #maybe look into tokenizination libraries (spacy, etc) 
    #Tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    train_texts_tok = [Tokenizer(t) for t in train_texts]
    test_texts_tok  = [Tokenizer(t) for t in test_texts]
    #we need to store the tokenized texts anyways 
    #tokenizer: string -> :list[:string] | splits on " " char 

    #####################
    #    build vocab    #
    #####################
    vecs = FastText('simple') #[CONFIG] following should be stored in a config file 
    
    voc = vocab({w:1 for w in vecs.itos})
    unk_idx = len(voc)-1
    voc.set_default_index(unk_idx)
    vecs.vectors[unk_idx] = vecs.vectors.mean(0)

    embed = torch.nn.Embedding.from_pretrained(vecs.vectors)

    """
    brief explanation of what's going on for my friends:
    - tokenizer: splits (cleaned/normalized) sentences into words
    - vocab: replace words by a number
      -> where do we get this number from? the fastText library, pre-trained
    - unk is just the encoding for words we don't know, i.e. don't see early on
    when building the vocab
    - embedding turns these numbers into vectors
      -> what number becomes what vector is also determined by fasttext
    """

    #############################
    #    back to prepro loop    #
    #############################

    #print("traintexttok[0]: ", train_texts_tok[0], "\n len: ", len(train_texts_tok[0]))
    #print("traintexttok0: ",train_texts_tok[0])

    train_texts_i= [embed(torch.tensor([voc[i] for i in t])) for t in train_texts_tok]
    test_texts_i = [embed(torch.tensor([voc[i] for i in t])) for t in test_texts_tok]   

    train_texts_padded = pad_sequence(train_texts_i, batch_first=True)
    test_texts_padded  = pad_sequence(test_texts_i, batch_first=True)

    train_data= [ [text, train_labels[i]] for i,text in enumerate(train_texts_padded)]
    test_data = [ [text, test_labels[i]] for i,text in enumerate(test_texts_padded)]

    batch_size=64 # (CONFIG FILE !!)

    cutoff_test = len(train_texts_i) % batch_size
    cutoff_train= len(test_texts_i) % batch_size
    ###################
    #   dataloader    #
    ###################

    #print("train data: ",type(train_data))

    train = DataLoader(train_data[:-cutoff_train], shuffle=True, batch_size=batch_size)
    test = DataLoader(test_data[:-cutoff_test], shuffle=True, batch_size=batch_size)
    print("data successfully loaded\n") 
    return train, test

>>>>>>> d40f8877301d07953316a013e22163936b2668e5
