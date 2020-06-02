import os
import glob


def mkpath(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path + '/'


def get_epoch(x):
    return int(x.split('.')[1].split('_')[2])


def get_last_epoch(weight_path):
    weights = sorted(glob.glob(os.path.join(
        weight_path, 'weight.*.h5')), key=get_epoch)
    last_weight = weights[-1] if weights else None
    last_epoch = get_epoch(last_weight) if last_weight else 0
    return last_weight, last_epoch

