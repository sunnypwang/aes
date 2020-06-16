import numpy as np
import pandas as pd
import os
from keras.callbacks import *
from sklearn.metrics import cohen_kappa_score
import utils
from data_utils import gen, prepare_features, rescale_to_int


class EvaluateCallback(Callback):
    def __init__(self, prompt, val_data, model_name, vocab=None, batch_size=5):
        self.prompt = prompt
        self.val_data = val_data
        self.model_name = model_name
        self.vocab = vocab
        self.batch_size = batch_size
        self.steps = np.ceil(len(val_data) / batch_size)
        self.y_true = prepare_features(model_name,
                                       df=val_data, prompt=prompt, y_only=True)

    def on_epoch_end(self, epoch, logs):
        y_pred = self.model.predict_generator(
            gen(self.model_name, self.prompt, self.val_data, self.vocab, self.batch_size, test=True, shuffle=False), steps=self.steps, verbose=1)

        generate_qwk(self.prompt, self.model_name,
                     self.y_true, y_pred, epoch+1, 'val')


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


def generate_summary(model_name, epoch):
    prompts = [1, 2, 3, 4, 5, 6, 7, 8]
    # number of essay in test set
    length = [-1, 179, 180, 173, 177, 181, 180, 157, 73]
    path = utils.mkpath('pred/{}'.format(model_name))

    with open(os.path.join(path, 'summary_{}.txt'.format(epoch)), 'w+') as f:
        f.write('{} epoch {}\n\n'.format(model_name, epoch))
        f.write('QWK\n')
        qwk_avg = 0
        for p in prompts:
            qwk_df = pd.read_csv(os.path.join(path, 'qwk_{}_test.csv'.format(
                p)), header=None, names=['epoch', 'qwk'])
            qwk = qwk_df[qwk_df['epoch'] == epoch].values[-1, -1]
            f.write('{}\t{}\n'.format(p, qwk))
            qwk_avg += qwk

        f.write('\nRobustness per prompt\n')
        r_avg = 0
        r_aug_avg = 0
        for p in prompts:
            robustness_df = pd.read_csv(os.path.join(
                path, 'robustness_{}_{}.csv'.format(p, epoch)))
            r = (robustness_df['worse_resolved'] -
                 robustness_df['better_resolved']).values[-1]
            f.write('{}\t{}\n'.format(p, r))
            r_avg += r

            r_aug = (robustness_df['worse_resolved'] -
                     robustness_df['better_resolved']).values[:-2]/length[p]
            r_aug_avg += r_aug

        f.write('\nRobustness per augment\n')
        r_aug_avg /= 8
        for a, r in zip(robustness_df['augment'][:-2], r_aug_avg):
            f.write('{}\t{}\n'.format(a, r))

        f.write('\n')
        f.write('QWK Average:\t{}\n'.format(qwk_avg / 8))
        f.write('Robustness Average:\t{}\n'.format(r_avg / 8))
        f.write('Robustness Average:\t{}\n'.format(r_aug_avg.mean()))
    print('summary generated!')


def generate_summary_best(model_name):
    prompts = [1, 2, 3, 4, 5, 6, 7, 8]
    # number of essay in test set
    length = [-1, 179, 180, 173, 177, 181, 180, 157, 73]
    path = utils.mkpath('pred/{}'.format(model_name))

    best_ep = [-1]*9
    with open(os.path.join(path, 'summary_best.txt'), 'w+') as f:
        f.write('{}\n\n'.format(model_name))
        f.write('QWK\n')
        f.write('epoch\tprompt\tqwk\n')
        qwk_avg = 0
        for p in prompts:
            qwk_df = pd.read_csv(os.path.join(path, 'qwk_{}_test.csv'.format(
                p)), header=None, names=['epoch', 'qwk'])
            max_idx = qwk_df['qwk'].idxmax()
            ep, qwk = qwk_df.iloc[max_idx].values
            best_ep[p] = int(ep)
            f.write('{}\t{}\t{}\n'.format(best_ep[p], p, qwk))
            qwk_avg += qwk

        f.write('\nRobustness per prompt\n')
        r_avg = 0
        r_aug_avg = 0
        for p in prompts:
            robustness_df = pd.read_csv(os.path.join(
                path, 'robustness_{}_{}.csv'.format(p, best_ep[p])))
            r = (robustness_df['worse_resolved'] -
                 robustness_df['better_resolved']).values[-1]
            f.write('{}\t{}\n'.format(p, r))
            r_avg += r

            r_aug = (robustness_df['worse_resolved'] -
                     robustness_df['better_resolved']).values[:-2]/length[p]
            r_aug_avg += r_aug

        f.write('\nRobustness per augment\n')
        r_aug_avg /= 8
        for a, r in zip(robustness_df['augment'][:-2], r_aug_avg):
            f.write('{}\t{}\n'.format(a, r))

        f.write('\n')
        f.write('QWK Average:\t{}\n'.format(qwk_avg / 8))
        f.write('Robustness Average:\t{}\n'.format(r_avg / 8))
        f.write('Robustness Average:\t{}\n'.format(r_aug_avg.mean()))
    print('summary generated!')


def QWK(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def robustness(original, augment, original_int, augment_int, threshold=0.0):
    worse_raw = np.sum(original - augment > threshold)
    better_raw = np.sum(augment - original > threshold)
    worse_resolved = np.sum(original_int > augment_int)
    better_resolved = np.sum(original_int < augment_int)
    return worse_raw, better_raw, worse_resolved, better_resolved
