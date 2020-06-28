import eval_utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, help='model name for path handling')
parser.add_argument('--epoch', type=int, default=0)
args = parser.parse_args()

EPOCH = args.epoch
MODEL_NAME = args.name

if EPOCH == 0:
    print('Best epoch mode')
    eval_utils.generate_summary_best(MODEL_NAME)
else:
    eval_utils.generate_summary(MODEL_NAME, EPOCH)
