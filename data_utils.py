import numpy as np
import pandas as pd
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from keras.preprocessing.sequence import pad_sequences

import utils
import eval_utils

augment_set = ['no_art', 'no_conj', 'add_and_0.1', 'swap_0.05',
               'no_first_sent', 'no_last_sent', 'no_longest_sent', 'reverse_sent']

MAXLEN = [-1, 70, 88, 22, 23, 24, 20, 67, 97]

PAD_SENT_TOKEN = ''


def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove quotation
    text = re.sub(r'\"', '', text)
    # URL replace
    text = re.sub(
        '(http[s]?://)?((www)\.)?([a-zA-Z0-9]+)\.{1}((com)(\.(cn))?|(org))', '<url>', text)
    # Truncate any duplicate non-alphanumeric and add a space after it
    # e.g. sent1.sent2!!!...??? becomes sent1. sent2! . ?
    text = re.sub(r'([^a-zA-Z0-9_@\'\s])\1*', r'\1 ', text)

    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    return text


def tokenize_text(text):
    '''Word tokenize using NLTK word_tokenize'''
    tokens = word_tokenize(text)
    for index, token in enumerate(tokens):
        if token == '@' and (index+1) < len(tokens):
            tokens[index+1] = '@' + tokens[index+1]
            tokens.pop(index)
    return tokens


def sentenize(text):
    '''Sentence tokenize using NLTK sent_tokenize'''
    sents = sent_tokenize(text)
    return sents


# def compute_maxsen(df, prompt):
#     maxx = 0
#     for essay in df['essay']:
#         essay = sentenize(essay)
#         if MAX_LEN[prompt] < len(essay):
#             MAX_LEN[prompt] = len(essay)


def prepare_elmo_features(df, prompt, features='essay', labels='domain1_score'):
    X_tmp = []
    for essay in df[features]:
        X_tmp.append(sentenize(essay))
    X_tmp = np.array(X_tmp)
    X_pad = pad_sequences(X_tmp, maxlen=MAXLEN[prompt], dtype=object,
                          padding='post', value=PAD_SENT_TOKEN)[:, :, None]
    Y = np.array(df[labels].tolist())
    Y_norm = eval_utils.normalize_score(Y, prompt)
    return X_pad, Y_norm


def load_data(prompt, suffix=None, fold=1):
    if suffix:
        data = pd.read_csv(
            'asap/fold_{}/prompt_{}_{}.tsv'.format(fold, prompt, suffix), sep='\t')
    else:
        data = pd.read_csv(
            'asap/fold_{}/prompt_{}.tsv'.format(fold, prompt), sep='\t')
    return data


def load_elmo_features(prompt, suffix=None, fold=1):
    data = load_data(prompt, suffix, fold)
    X, Y = prepare_elmo_features(data, prompt)
    return X, Y


def elmo_gen(prompt, suffix, fold=1, batch_size=1):
    data = load_data(prompt, suffix, fold)
    while True:
        for i in range(0, len(data), batch_size):
            j = min(len(data), i+batch_size)
            x, y = prepare_elmo_features(data[i:j], prompt)
            yield x, y


# def load_data(prompt_id, fold_id, suffix):
#     path = 'prompt_{}/new/fold_{}/prompt_{}_{}.tsv'.format(
#         prompt_id, fold_id, prompt_id, suffix)
#     df = pd.read_csv(path, sep='\t')
#     return df


def clean_data(df):
    new_df = []
    for essay in df:
        new_df.append(clean_text(essay))
    return new_df


