import numpy as np
from keras.callbacks import *
from sklearn.metrics import cohen_kappa_score
import utils

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


class EvaluateCallback(Callback):
    def __init__(self, x_val, y_val, model_name, prompt, batch_size=5):
        self.x_val = x_val
        self.y_val = y_val
        self.prompt = prompt
        self.batch_size = batch_size
        self.model_name = model_name

    def on_epoch_end(self, epoch, logs):
        y_pred = self.model.predict(self.x_val, batch_size=self.batch_size)
        y_pred = rescale_to_int(self.y_pred, self.prompt)
        y_true = rescale_to_int(self.y_val, self.prompt)

        evaluate(y_true, y_pred, self.model_name, self.prompt, epoch)


def evaluate(y_true, y_pred, model_name, prompt, epoch):
    qwk = QWK(y_true, y_pred)
    pred_path = utils.mkpath('pred')
    with open(os.path.join(pred_path, 'qwk_{}_{}.txt'.format(model_name, prompt)), 'a+') as f:
        f.write('{} {}'.format(epoch, qwk))


def QWK(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def robustness(original, augment, original_int, augment_int, threshold=0.0):
    worse_raw = np.sum(original - augment > threshold)
    better_raw = np.sum(augment - original > threshold)
    worse_resolved = np.sum(original_int > augment_int)
    better_resolved = np.sum(original_int < augment_int)
    return worse_raw, better_raw, worse_resolved, better_resolved
