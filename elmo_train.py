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
parser.add_argument('--bs', type=int, default=5)
parser.add_argument('--fold', type=int, default=1)
parser.add_argument('--gen-augment', type=bool, default=False)
args = parser.parse_args()

prompts = [args.prompt]
if args.prompt == -1:
    prompts = [1, 2, 3, 4, 5, 6, 7, 8]

print(args)
print('PROMPT :', prompts)

BATCH_SIZE = args.bs
MODEL_NAME = 'elmo'

for p in prompts:

    train_df = data_utils.load_data(p, 'train')[:10]
    val_df = data_utils.load_data(p, 'val')[:10]
    # test_df = data_utils.load_data(p, 'test')

    print(train_df.shape)
    print(val_df.shape)
    # print(test_df.shape)

    from keras import backend as K
    K.clear_session()
    model = models.build_elmo_model_full(p)

    weight_path = utils.mkpath('weight')
    last_weight, last_epoch = utils.get_last_epoch(weight_path, MODEL_NAME, p)
    if last_weight:
        model.load_weights(last_weight)
    assert args.epoch > last_epoch

    train_gen = data_utils.elmo_gen(p, train_df, batch_size=BATCH_SIZE)
    val_gen = data_utils.elmo_gen(p, val_df, batch_size=BATCH_SIZE)

    train_steps = np.ceil(len(train_df) / BATCH_SIZE)
    val_steps = np.ceil(len(val_df) / BATCH_SIZE)

    print(train_steps, val_steps)

    callbacks = [ModelCheckpoint(os.path.join(weight_path, 'weight.{}_{}_{{epoch:02d}}_{{val_loss:.4f}}.h5'.format(MODEL_NAME, p)), save_weights_only=True, period=1),
                 CSVLogger(os.path.join(
                     weight_path, 'history.csv'), append=True),
                 eval_utils.EvaluateCallback(p, val_df, MODEL_NAME, BATCH_SIZE)]
    model.fit_generator(train_gen, steps_per_epoch=train_steps,
                        validation_data=val_gen, validation_steps=val_steps,
                        epochs=args.epoch, initial_epoch=last_epoch,
                        callbacks=callbacks)
