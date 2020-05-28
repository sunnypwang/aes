import sys
import argparse
from keras.callbacks import *


import utils
import data_utils
import eval_utils
import models

parser = argparse.ArgumentParser()
parser.add_argument('prompt', type=int, help='-1 for all prompts')
parser.add_argument('epoch', type=int)
parser.add_argument('--fold', type=int, default=1)
parser.add_argument('--gen-augment', type=bool, default=False)
args = parser.parse_args()

prompts = [args.prompt]
if args.prompt == -1:
    prompts = [1, 2, 3, 4, 5, 6, 7, 8]

print(args)
print('PROMPT :', prompts)

BATCH_SIZE = 1
MODEL_NAME = 'elmo'

for p in prompts:

    X_train, Y_train = data_utils.load_elmo_features(p, 'train')
    X_val, Y_val = data_utils.load_elmo_features(p, 'val')
    X_test, Y_test = data_utils.load_elmo_features(p, 'test')

    print(X_train.shape, Y_train.shape)
    print(X_val.shape, Y_val.shape)
    print(X_test.shape, Y_test.shape)

    model = models.build_elmo_model_full(p)

    weight_path = utils.mkpath('weight')
    last_weight, last_epoch = utils.get_last_epoch(weight_path, MODEL_NAME, p)
    if last_weight:
        model.load_weights(last_weight)
    assert args.epoch > last_epoch

    # train_gen = data_utils.elmo_gen(p, 'train', batch_size=BATCH_SIZE)
    # val_gen = data_utils.elmo_gen(p, 'val', batch_size=BATCH_SIZE)

    # train_steps = 1
    # val_steps = 1

    callbacks = [ModelCheckpoint(os.path.join(weight_path, 'weight.{}_{}_{{epoch:02d}}_{{val_loss:.4f}}.h5'.format(MODEL_NAME, p)), save_weights_only=True, period=1),
                 CSVLogger(os.path.join(
                     weight_path, 'history.csv'), append=True),
                 eval_utils.EvaluateCallback(X_val, Y_val, MODEL_NAME, p, BATCH_SIZE)]
    model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=args.epoch,
              initial_epoch=last_epoch, callbacks=callbacks, validation_data=(X_val, Y_val))
