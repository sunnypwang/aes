import sys
import argparse

import os
import numpy as np
from keras import backend as K
from keras.callbacks import *

import utils
import data_utils
import eval_utils
import models

parser = argparse.ArgumentParser()
parser.add_argument('prompt', type=int, help='-1 for all prompts')
parser.add_argument('epoch', type=int)
parser.add_argument('name', type=str, help='model name for path handling')
parser.add_argument('--bs', type=int, default=5)
parser.add_argument('--fold', type=int, default=1)
parser.add_argument('--ft', type=bool, default=False,
                    help='enable fine-tuning')
parser.add_argument('--augment', type=bool, default=True,
                    help='include augment during testing')
args = parser.parse_args()

prompts = [args.prompt]
if args.prompt == -1:
    prompts = [1, 2, 3, 4, 5, 6, 7, 8]

EPOCH = args.epoch
BATCH_SIZE = args.bs
MODEL_NAME = args.name

print(args)
print('ALL PROMPTS :', prompts)
print('BATCH SIZE :', BATCH_SIZE)
print('MODEL_NAME :', MODEL_NAME)
print('EPOCH :', EPOCH)
print('-------')

for p in prompts:
    print('PROMPT :', p)

    weight_path = utils.mkpath('weight/{}/{}'.format(MODEL_NAME, p))
    weight = utils.get_weight_at_epoch(weight_path, EPOCH)
    if not weight:
        print('weight not found')
        continue

    test_df = data_utils.load_data(p, 'test')

    print(test_df.shape)

    K.clear_session()
    if MODEL_NAME.startswith('elmo'):
        vocab = None
        model = models.build_elmo_model_full(
            p,  only_elmo=False, use_mask=True, summary=False)
    elif MODEL_NAME.startswith('glove'):
        vocab = data_utils.get_vocab(p)
        glove_path = 'glove/glove.6B.50d.txt'
        emb_matrix = data_utils.load_glove_embedding(glove_path, vocab)
        model = models.build_glove_model(
            p, len(vocab), emb_matrix, glove_trainable=False, summary=False)

    print('Loading weight :', weight)
    model.load_weights(weight)

    test_gen = data_utils.gen(MODEL_NAME,
                              p, test_df, vocab, batch_size=BATCH_SIZE, test=True, shuffle=False)

    test_steps = np.ceil(len(test_df) / BATCH_SIZE)

    print(test_gen, test_steps)

    y_true = data_utils.prepare_features(MODEL_NAME,
                                         df=test_df, prompt=p, vocab=vocab, y_only=True, norm=True)

    y_pred = model.predict_generator(
        test_gen, steps=test_steps, verbose=1)

    eval_utils.generate_qwk(p, MODEL_NAME, y_true,
                            y_pred, EPOCH, suffix='test')

    if args.augment:
        print('Predicting on augment sets...')
        aug_pred = {}
        for augment in data_utils.augment_set:
            aug_gen = data_utils.augment_gen(MODEL_NAME,
                                             p, test_df, vocab=vocab, batch_size=BATCH_SIZE, augment=augment)
            aug_steps = np.ceil(len(test_df) / BATCH_SIZE)

            aug_pred[augment] = model.predict_generator(
                aug_gen, steps=aug_steps, verbose=1)

        eval_utils.generate_score(
            p, MODEL_NAME, EPOCH, y_true, y_pred, aug_pred, test_df)

        eval_utils.generate_robustness(
            p, MODEL_NAME, EPOCH, y_true, y_pred, aug_pred)

if len(prompts) == 8:
    eval_utils.generate_summary(MODEL_NAME, EPOCH)
