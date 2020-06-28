import numpy as np
import pandas as pd
import re
from nltk.tokenize import sent_tokenize, word_tokenize
import json
import collections
from keras.preprocessing.sequence import pad_sequences

import os

import utils

augment_set = ['no_art', 'no_conj', 'add_and-0.1', 'swap_word-0.05',
               'no_first_sent', 'no_last_sent', 'no_longest_sent', 'reverse_sent']


MAXLEN = [-1, 70, 88, 22, 23, 24, 20, 67, 97]     # Max
# MAXLEN = [-1, 47, 44, 14, 10, 15, 16, 32, 79]       # 1.5IQR Max
MAXWORDLEN = 50

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


def tokenize(text):
    '''Word tokenize using NLTK word_tokenize'''
    tokens = word_tokenize(text)
    for index, token in enumerate(tokens):
        if token == '@' and (index+1) < len(tokens):
            tokens[index+1] = '@' + re.sub('[0-9]+.*', '', tokens[index+1])
            tokens.pop(index)
    return tokens


def sentenize(text):
    '''Sentence tokenize using NLTK sent_tokenize'''
    sents = sent_tokenize(text)
    return sents


def shorten_sentence(tokens):
    if len(tokens) <= MAXWORDLEN:
        return [tokens]

    # Step 1: split sentence based on keywords
    # split_keywords = ['because', 'but', 'so', 'then', 'You', 'He', 'She', 'We', 'It', 'They', 'Your', 'His', 'Her']
    split_keywords = ['because', 'but', 'so', 'then']
    k_indexes = [i for i, key in enumerate(tokens) if key in split_keywords]
    processed_tokens = []
    if not k_indexes:
        num = len(tokens) // MAXWORDLEN
        k_indexes = [(i+1)*MAXWORDLEN for i in range(num)]

    if len(tokens[:k_indexes[0]]) > 0:
        processed_tokens.append(tokens[:k_indexes[0]])
    len_k = len(k_indexes)
    for j in range(len_k-1):
        processed_tokens.append(tokens[k_indexes[j]:k_indexes[j+1]])
    processed_tokens.append(tokens[k_indexes[-1]:])

    # Step 2: split sentence to no more than MAXWORDLEN
    # if there are still sentences whose length exceeds MAXWORDLEN
    new_tokens = []
    for token in processed_tokens:
        if len(token) > MAXWORDLEN:
            num = len(token) // MAXWORDLEN
            s_indexes = [(i+1)*MAXWORDLEN for i in range(num)]
            len_s = len(s_indexes)
            if len(token[:s_indexes[0]]) > 0:
                new_tokens.append(token[0:s_indexes[0]])
            for j in range(len_s-1):
                new_tokens.append(token[s_indexes[j]:s_indexes[j+1]])
            new_tokens.append(token[s_indexes[-1]:])
        else:
            new_tokens.append(token)
    # print('before', len(tokens), 'after', [len(x) for x in new_tokens])
    return new_tokens


def load_glove_embedding(path, vocab, emb_dim=50):
    scale = np.sqrt(3.0 / emb_dim)
    emb_matrix = np.empty((len(vocab), emb_dim))
    emb_dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            args = line.split()
            word = args[0]
            vec = args[1:]
            emb_dict[word] = vec
    oov = 0
    for w in vocab:
        if w in emb_dict:
            emb = np.array(emb_dict[w])
        else:
            emb = np.random.uniform(-scale, scale, emb_dim)
            oov += 1
        emb_matrix[vocab[w]] = emb
    print('OOV: ', oov/len(vocab))
    del emb_dict
    return emb_matrix


def get_vocab(prompt, df=None, length=4000, features='essay'):
    vocab_path = utils.mkpath('vocab')
    file_path = os.path.join(vocab_path, '{}.vocab'.format(prompt))
    if os.path.isfile(file_path):
        with open(file_path, 'r') as f:
            vocab = json.load(f)
        assert type(vocab) == dict
        print('load vocab from {}'.format(file_path))
        return vocab

    word_all = []
    for essay in df[features]:
        sents = sentenize(essay)
        for sent in sents:
            words = tokenize(sent)
            word_all.extend(words)
    print('word count:', len(word_all))
    print('unique word count:', len(set(word_all)))

    most_common = collections.Counter(word_all).most_common(length - 3)

    vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2}
    for w, c in most_common:
        vocab[w] = len(vocab)

    # save as JSON
    with open(file_path, 'w') as f:
        json.dump(vocab, f)
    print('save vocab to {}'.format(file_path))

    return vocab


# def compute_maxsen(df, prompt):
#     maxx = 0
#     for essay in df['essay']:s
#         if MAX_LEN[prompt] < len(essay):
#             MAX_LEN[prompt] = len(essay)

def word2idx(w, vocab):
    if not w in vocab:
        return vocab['<unk>']
    return vocab[w]


def prepare_glove_features(df, prompt, vocab=None, features='essay', labels='domain1_score', x_only=False, pad=True, split_long_sent=True, y_only=False, norm=True, augment=None, rnd=None):
    assert not (x_only and y_only)
    if not y_only:
        X = np.zeros((len(df), MAXLEN[prompt], MAXWORDLEN), dtype=int)
        if not vocab:
            vocab = get_vocab(prompt, df)
        for i, essay in enumerate(df[features]):
            sents = sentenize(essay)
            if augment:
                sents = make_augment(sents, augment, rnd)
            sent_idxs = []
            for sent in sents:
                words = tokenize(sent)
                if split_long_sent:
                    split_list = shorten_sentence(words)
                    for word_tokens in split_list:
                        sent_idxs.append([word2idx(w, vocab)
                                          for w in word_tokens])
                else:
                    sent_idxs.append([word2idx(w, vocab) for w in words])

            if pad:
                sent_idxs = pad_sequences(
                    sent_idxs, maxlen=MAXWORDLEN, dtype=object, padding='post', truncating='post', value=0)
            # print(sent_idxs.shape == X[i, :len(sent_idxs)].shape)
            X[i, :len(sent_idxs)] = sent_idxs
        if x_only:
            return X
    if not x_only:
        Y = np.array(df[labels].tolist())
        if norm:
            Y = normalize_score(Y, prompt)
        if y_only:
            return Y
    return X, Y


def prepare_elmo_features(df, prompt, vocab=None, features='essay', labels='domain1_score', x_only=False, pad=True, y_only=False, norm=True, augment=None, rnd=None):
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


def prepare_features(model_name, **kwargs):
    if model_name.startswith('elmo'):
        return prepare_elmo_features(**kwargs)
    elif model_name.startswith('glove'):
        return prepare_glove_features(**kwargs)


def gen(model_name, prompt, df, vocab=None, batch_size=1, test=False, shuffle=True, **kwargs):
    data = df.copy()
    while True:
        if shuffle:
            data = data.sample(frac=1).reset_index(drop=True)
        for i in range(0, len(data), batch_size):
            j = min(len(data), i+batch_size)
            if test:
                x = prepare_features(model_name,
                                     df=data[i:j], prompt=prompt, vocab=vocab, x_only=True, **kwargs)
                yield x
            else:
                x, y = prepare_features(model_name,
                                        df=data[i:j], prompt=prompt, vocab=vocab, **kwargs)
                yield x, y


def elmo_gen(prompt, df, batch_size=1, test=False, shuffle=True, **kwargs):
    data = df.copy()
    while True:
        if shuffle:
            data = data.sample(frac=1).reset_index(drop=True)
        for i in range(0, len(data), batch_size):
            j = min(len(data), i+batch_size)
            if test:
                x = prepare_elmo_features(
                    data[i:j], prompt, x_only=True, **kwargs)
                yield x
            else:
                x, y = prepare_elmo_features(data[i:j], prompt, **kwargs)
                yield x, y


def augment_gen(model_name, prompt, test_df, vocab=None, batch_size=1, augment=None, **kwargs):
    data = test_df.copy()
    rnd = np.random.RandomState(1)
    while True:
        for i in range(0, len(data), batch_size):
            j = min(len(data), i+batch_size)

            x = prepare_features(model_name,
                                 df=data[i:j], prompt=prompt, vocab=vocab, x_only=True, augment=augment, rnd=rnd, **kwargs)
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
