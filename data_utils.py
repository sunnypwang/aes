import numpy as np
import pandas as pd
import re
from nltk.tokenize import sent_tokenize, word_tokenize

import utils

augment_set = ['no_art', 'no_conj', 'add_and-0.1', 'swap_word-0.05',
               'no_first_sent', 'no_last_sent', 'no_longest_sent', 'reverse_sent']


# MAXLEN = [-1, 70, 88, 22, 23, 24, 20, 67, 97]     # Max
MAXLEN = [-1, 47, 44, 14, 10, 15, 16, 32, 79]       # 1.5IQR Max

PAD_SENT_TOKEN = ''

score_range = [(-1, -1),
               (2, 12),
               (1, 6),
               (0, 3),
               (0, 3),
               (0, 4),
               (0, 4),
               (0, 30),
               (0, 60)]


def get_threshold(p):
    low, high = score_range[p]
    return 1/((high - low))


def rescale_to_int(raw, p):
    assert (raw >= 0.).all() and (raw <= 1.).all()
    low, high = score_range[p]
    return np.around(raw * (high - low) + low).astype(int)


def normalize_score(Y, p):
    low, high = score_range[p]
    Y = np.array(Y)
    Y_norm = (Y - low)/(high - low)
    assert (Y_norm >= 0.).all() and (Y_norm <= 1.).all()
    Y_resolved = rescale_to_int(Y_norm, p)
    try:
        assert np.equal(Y, Y_resolved).all()
    except AssertionError:
        for i in range(len(Y)):
            if Y[i] != Y_resolved[i]:
                print(i, Y[i], Y_resolved[i])
        print('use python3')
    return Y_norm


def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove quotation
    text = re.sub(r'\"', '', text)
    # URL replace by https://github.com/feidong1991/aes
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


def prepare_elmo_features(df, prompt, features='essay', labels='domain1_score', x_only=False, pad=True, y_only=False, norm=True, augment=None, rnd=None):
    assert not (x_only and y_only)
    if not y_only:
        X = []
        # Sentence tokenize and make augment
        for essay in df[features]:
            sents = sentenize(essay)
            if augment:
                X.append(make_augment(sents, augment, rnd))
            else:
                X.append(sents)
        X = np.array(X)
        if pad:
            from keras.preprocessing.sequence import pad_sequences
            X = pad_sequences(X, maxlen=MAXLEN[prompt], dtype=object,
                              padding='post', truncating='post', value=PAD_SENT_TOKEN)[:, :, None]
        if x_only:
            return X
    if not x_only:
        Y = np.array(df[labels].tolist())

        if norm:
            Y = normalize_score(Y, prompt)

        if y_only:
            return Y

    return X, Y


def load_data(prompt, suffix=None, fold=1):
    if suffix:
        data = pd.read_csv(
            'asap/fold_{}/prompt_{}_{}.tsv'.format(fold, prompt, suffix), sep='\t')
    else:
        data = pd.read_csv(
            'asap/fold_{}/prompt_{}.tsv'.format(fold, prompt), sep='\t')
    return data


def load_elmo_features(prompt, suffix=None, fold=1, **kwargs):
    data = load_data(prompt, suffix, fold)
    return prepare_elmo_features(
        data, prompt, **kwargs)


def elmo_gen(prompt, df, batch_size=1, test=False, **kwargs):
    data = df.copy()
    while True:
        for i in range(0, len(data), batch_size):
            j = min(len(data), i+batch_size)
            if test:
                x = prepare_elmo_features(
                    data[i:j], prompt, x_only=True, **kwargs)
                yield x
            else:
                x, y = prepare_elmo_features(data[i:j], prompt, **kwargs)
                yield x, y


def augment_gen(prompt, test_df, batch_size=1, augment=None, **kwargs):
    data = test_df.copy()
    rnd = np.random.RandomState(1)
    while True:
        for i in range(0, len(data), batch_size):
            j = min(len(data), i+batch_size)

            x = prepare_elmo_features(
                data[i:j], prompt, x_only=True, augment=augment, rnd=rnd, **kwargs)
            yield x


def make_augment(sents, augment, rnd=None):
    '''augment essay (list of sentences)'''
    assert augment in augment_set
    t = augment.split('-')
    if len(t) > 1:
        augment, threshold = t[0], float(t[1])
    else:
        threshold = 1.0

    new_sents = []
    if not rnd:
        rnd = np.random.RandomState(1)

    if augment == 'no_art':
        for sent in sents:
            new_sents.append(re.sub(r'\b(a|an|the)\b ', r'', sent))

    elif augment == 'no_conj':
        for sent in sents:
            new_sents.append(re.sub(r'\b(and|or|but)\b ', r'', sent))

    elif augment == 'add_and':
        for sent in sents:
            state = rnd.rand()
            if state < threshold:
                sent = 'and ' + sent
            new_sents.append(sent)

    elif augment == 'swap_word':
        for sent in sents:
            words = sent.split()
            word_idx = np.arange(len(words)-2)
            rnd.shuffle(word_idx)
            for i in word_idx:
                state = rnd.rand()
                if state < threshold:
                    words[i], words[i+1] = words[i+1], words[i]
            new_sents.append(' '.join(words))

    elif augment == 'no_first_sent':
        if len(sents) > 1:
            new_sents.extend(sents[1:])
        else:
            new_sents.extend(['.'])

    elif augment == 'no_last_sent':
        if len(sents) > 1:
            new_sents.extend(sents[:-1])
        else:
            new_sents.extend(['.'])

    elif augment == 'no_longest_sent':
        if len(sents) > 1:
            maxidx = np.argmax([len(sent) for sent in sents])
            new_sents.extend(sents[:maxidx] + sents[maxidx+1:])
        else:
            new_sents.extend(['.'])

    elif augment == 'reverse_sent':
        new_sents.extend(sents[::-1])

    else:
        raise NameError('Unknown augment : ' + str(augment))
    assert type(new_sents) is list
    return new_sents

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
