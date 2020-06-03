import numpy as np
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


def QWK(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def robustness(original, augment, original_int, augment_int, threshold=0.0):
    worse_raw = np.sum(original - augment > threshold)
    better_raw = np.sum(augment - original > threshold)
    worse_resolved = np.sum(original_int > augment_int)
    better_resolved = np.sum(original_int < augment_int)
    return worse_raw, better_raw, worse_resolved, better_resolved
