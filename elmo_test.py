import sys
import argparse
from keras.callbacks import *
import os
import numpy as np

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
                    help='enable fine-tuning ELMo (elno_trainable)')
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

    from keras import backend as K
    K.clear_session()
    model = models.build_elmo_model_full(
        p,  only_elmo=False, use_mask=True, summary=False)

    print('Loading weight :', weight)
    model.load_weights(weight)

    test_gen = data_utils.elmo_gen(p, test_df, batch_size=BATCH_SIZE)
    test_steps = np.ceil(len(test_df) / BATCH_SIZE)

    print(test_gen, test_steps)

    y_true = data_utils.prepare_elmo_features(
        test_df, p, y_only=True, norm=True)

    y_pred = model.predict_generator(
        test_gen, steps=test_steps, verbose=1)

    eval_utils.generate_qwk(p, MODEL_NAME, y_true,
                            y_pred, EPOCH, suffix='test')

    if args.augment:
        print('Predicting on augment sets...')
        aug_pred = {}
        for augment in data_utils.augment_set:
            aug_gen = data_utils.augment_gen(
                p, test_df, batch_size=BATCH_SIZE, augment=augment)
            aug_steps = np.ceil(len(test_df) / BATCH_SIZE)

            aug_pred[augment] = model.predict_generator(
                aug_gen, steps=aug_steps, verbose=1)

        eval_utils.generate_score(
            p, MODEL_NAME, EPOCH, y_true, y_pred, aug_pred, test_df)

        eval_utils.generate_robustness(
            p, MODEL_NAME, EPOCH, y_true, y_pred, aug_pred)
