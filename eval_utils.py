import numpy as np
import pandas as pd
import os
from keras.callbacks import *
from sklearn.metrics import cohen_kappa_score
import utils
from data_utils import elmo_gen, prepare_elmo_features, rescale_to_int


class EvaluateCallback(Callback):
    def __init__(self, prompt, val_data, model_name, batch_size=5):
        self.prompt = prompt
        self.val_data = val_data
        self.model_name = model_name
        self.batch_size = batch_size
        self.steps = np.ceil(len(self.val_data) / self.batch_size)
        self.y_true = prepare_elmo_features(
            self.val_data, self.prompt, y_only=True)

    def on_epoch_end(self, epoch, logs):
        y_pred = self.model.predict_generator(
            elmo_gen(self.prompt, self.val_data, self.batch_size, test=True), steps=self.steps, verbose=1)

        generate_qwk(self.prompt, self.model_name,
                     self.y_true, y_pred, epoch, 'val')


def generate_qwk(prompt, model_name, y_true, y_pred, epoch, suffix=''):
    path = utils.mkpath('pred/{}'.format(model_name))

    y_true = rescale_to_int(y_true, prompt)
    y_pred = rescale_to_int(y_pred, prompt)
    qwk = QWK(y_true, y_pred)

    with open(os.path.join(path, 'qwk_{}_{}.csv'.format(prompt, suffix)), 'a+') as f:
        f.write('{}, {}\n'.format(epoch, qwk))


def generate_score(prompt, model_name, epoch, y_true, y_pred, aug_pred, test_df):
    path = utils.mkpath('pred/{}'.format(model_name))

    df = pd.DataFrame()
    df['essay_id'] = test_df['essay_id']
    df['essay_set'] = test_df['essay_set']
    df['domain1_score'] = y_true
    df['test'] = y_pred
    for key in aug_pred:
        df['test_' + key] = aug_pred[key]
    df.to_csv(os.path.join(path, 'score_{}_{}.tsv'.format(prompt, epoch)),
              sep='\t', index=False)
    return df


def generate_robustness(prompt, model_name, epoch, y_true, y_pred, aug_pred):
    path = utils.mkpath('pred/{}'.format(model_name))

    # y_true = rescale_to_int(y_true, prompt)
    y_pred_int = rescale_to_int(y_pred, prompt)
    aug_pred_int = {}
    wr_t, br_t, w_t, b_t = 0, 0, 0, 0
    N = len(y_pred) * len(aug_pred)
    print('N :', N)

    with open(os.path.join(path, 'robustness_{}_{}.csv'.format(prompt, epoch)), 'w+') as f:
        f.write('augment,worse_raw,better_raw,worse_resolved,better_resolved\n')
        for key in aug_pred:
            aug_pred_int[key] = rescale_to_int(aug_pred[key], prompt)

            wr, br, w, b = robustness(
                y_pred, aug_pred[key], y_pred_int, aug_pred_int[key])
            wr_t += wr
            br_t += br
            w_t += w
            b_t += b
            f.write('{},{},{},{},{}\n'.format(key, wr, br, w, b))
        f.write('sum,{},{},{},{}\n'.format(wr_t, br_t, w_t, b_t))
        f.write('avg,{},{},{},{}\n'.format(wr_t/N, br_t/N, w_t/N, b_t/N))


def QWK(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def robustness(original, augment, original_int, augment_int, threshold=0.0):
    worse_raw = np.sum(original - augment > threshold)
    better_raw = np.sum(augment - original > threshold)
    worse_resolved = np.sum(original_int > augment_int)
    better_resolved = np.sum(original_int < augment_int)
    return worse_raw, better_raw, worse_resolved, better_resolved
