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
parser.add_argument('--bs', type=int, default=10)
parser.add_argument('--fold', type=int, default=1)
parser.add_argument('--ft', action='store_true',
                    help='enable fine-tuning')
parser.add_argument('--re', type=int, default=100,
                    help='recurrent size (elmo)')
parser.add_argument('--drop', type=float, default=0.5,
                    help='dropout')
parser.add_argument('--mask', action='store_true')
args = parser.parse_args()

prompts = [args.prompt]
if args.prompt == -1:
    prompts = [1, 2, 3, 4, 5, 6, 7, 8]


BATCH_SIZE = args.bs
MODEL_NAME = args.name

print(args)
print('ALL PROMPTS :', prompts)
print('BATCH SIZE :', BATCH_SIZE)
print('MODEL_NAME :', MODEL_NAME)
print('-------')

for p in prompts:
    print('PROMPT :', p)

    weight_path = utils.mkpath('weight/{}/{}'.format(MODEL_NAME, p))
    last_weight, last_epoch = utils.get_last_epoch(weight_path)
    # move on to next prompt if epoch not greater than last one saved
    if args.epoch <= last_epoch:
        continue

    train_df = data_utils.load_data(p, 'train')
    val_df = data_utils.load_data(p, 'val')
    # test_df = data_utils.load_data(p, 'test')

    print(train_df.shape)
    print(val_df.shape)
    # print(test_df.shape)

    from keras import backend as K
    K.clear_session()
    model = models.build_elmo_model_full(
        p,  elmo_trainable=args.ft, use_mask=args.mask, lstm_units=args.re, drop_rate=args.drop)

    if last_weight:
        print('Loading weight :', last_weight)
        model.load_weights(last_weight)

    train_gen = data_utils.gen(
        MODEL_NAME, p, train_df, batch_size=BATCH_SIZE)
    val_gen = data_utils.gen(MODEL_NAME,
                             p, val_df, batch_size=BATCH_SIZE, shuffle=False)

    train_steps = np.ceil(len(train_df) / BATCH_SIZE)
    val_steps = np.ceil(len(val_df) / BATCH_SIZE)

    print(train_steps, val_steps)

    callbacks = [ModelCheckpoint(os.path.join(weight_path, 'weight.{}_{}_{{epoch:02d}}_{{val_loss:.4f}}.h5'.format(MODEL_NAME, p)), save_weights_only=True, period=1),
                 CSVLogger(os.path.join(
                     weight_path, 'history.csv'), append=True),
                 eval_utils.EvaluateCallback(p, val_df, MODEL_NAME, batch_size=BATCH_SIZE)]
    model.fit_generator(train_gen, steps_per_epoch=train_steps,
                        validation_data=val_gen, validation_steps=val_steps,
                        epochs=args.epoch, initial_epoch=last_epoch,
                        callbacks=callbacks)
