import bz2
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
#from torchtext.vocab import GloVe

##############################
#    read raw gros bizou2    #
##############################

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


def amzreview_dataset() -> (DataLoader, DataLoader):
    """
    reads files from /data/raw with appropriate method 
    applies normalization and tokenization
    merges processed texts with their labels into 
    torch.utils.data DataLoader object for train and test sets
    """
    train_labels, train_texts = get_labels_and_texts('data/raw/train.ft.txt.bz2')
    test_labels, test_texts = get_labels_and_texts('data/raw/test.ft.txt.bz2')

    train_texts = normalize_texts(train_texts)
    test_texts = normalize_texts(test_texts)
    #print("#txts mod 64: ", len(train_texts) % 64)
    #MAX_SEQ_SIZE = max([len(t) for t in train_texts+test_texts])


    ####################
    #    tokenize      # (already normalized by hand above)
    ####################
    Tokenizer = get_tokenizer(None) #maybe look into tokenizination libraries (spacy, etc) 
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

    train = DataLoader(train_data[:-cutoff_train], shuffle=True, batch_size=batch_size)
    test = DataLoader(test_data[:-cutoff_test], shuffle=True, batch_size=batch_size)
    return train, test

#amzreview_dataset()

