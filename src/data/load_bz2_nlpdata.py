import bz2
import os.path
import numpy as np
import re
from torchtext.data import get_tokenizer
from torch.utils.data import DataLoader


#from torchnlp.encoders.text import StaticTokenizerEncoder, stack_and_pad_tensors, pad_tensor


##############################
#    read raw gros bizou2    #
##############################

def get_labels_and_texts(file):
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

    train = DataLoader(train_data, shuffle=True, batch_size=3000)
    test = DataLoader(test_data, shuffle=True, batch_size=3000)
    return train, test

amzreview_dataset()


