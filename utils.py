import os
import glob


def mkpath(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path + '/'


def format_checkpoint(name, prompt):
    # Example : weight.elmo_1_50_0.1199.hdf5
    return 'weight.{}_{}_\{epoch:02d\}_\{val_loss:.4f\}.h5'.format(name, prompt)


def get_last_epoch(weight_path, name, prompt):
    def get_epoch(x): return int(x.split('.')[1].split('_')[2])
    weights = sorted(glob.glob(os.path.join(
        weight_path, 'weight.*.h5')), key=get_epoch)
    last_weight = weights[-1] if weights else None
    last_epoch = get_epoch(last_weight) if last_weight else 0
    return last_weight, last_epoch
