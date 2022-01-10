import bz2
import os.path
import numpy as np
import re

import torchtext
from torchtext.data import get_tokenizer
from torchtext.legacy.data import Field
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torchtext.vocab import FastText, voca
bimport torch
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

    #MAX_SEQ_SIZE = max([len(t) for t in train_texts+test_texts])


    ####################
    #    tokenize      # (already normalized by hand above)
    ####################
    Tokenizer = get_tokenizer(None) #maybe look into tokenizination libraries (spacy, etc) 
    train_texts_tok = [Tokenizer(t) for t in train_texts]
    test_texts_tok  = [Tokenizer(t) for t in test_texts]
    #we need to store the tokenized texts anyways 
    text_field = Field(tokenize=None, lower=True)
    #tokenizer: string -> :list[:string] | splits on " " char 


    #####################
    #    build vocab    #
    #####################
    text_field.build_vocab(train_texts_tok, 
        vectors='fasttext.simple.300d')# get the vocab instance
    vocab = text_field.vocab
    #print("vocab[are]: ", vocab['stuning'], "len(vocab): ", len(vocab))
    # load fasttext simple embedding with 300d
    embedding = FastText('simple') #[CONFIG] following should be stored in a config file 

    NUM_EMBEDDINGS = len(vocab)
    LEN_EMBEDDINGS = len(embedding[0])
    print("dim embeddings: {}\n num embeddings: {}".format(LEN_EMBEDDINGS, NUM_EMBEDDINGS))


    #############################
    #    back to prepro loop    #
    #############################

    #print("traintexttok[0]: ", train_texts_tok[0], "\n len: ", len(train_texts_tok[0]))

    train_texts_i= [torch.tensor([vocab[i] for i in t]) for t in train_texts_tok]
    test_texts_i = [torch.tensor([vocab[i] for i in t]) for t in test_texts_tok]   

    train_texts_padded = pad_sequence(train_texts_i, batch_first=True)
    test_texts_padded  = pad_sequence(test_texts_i, batch_first=True)

    train_data= [ [text, train_labels[i]] for i,text in enumerate(train_texts_padded)]
    test_data = [ [text, test_labels[i]] for i,text in enumerate(test_texts_padded)]

    ###################
    #   dataloader    #
    ###################

    train = DataLoader(train_data, shuffle=True, batch_size=64)
    test = DataLoader(test_data, shuffle=True, batch_size=64)
    return train, test

#amzreview_dataset()
