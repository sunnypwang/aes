import eval_utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('epoch', type=int)
parser.add_argument('name', type=str, help='model name for path handling')
args = parser.parse_args()

EPOCH = args.epoch
MODEL_NAME = args.name

eval_utils.generate_summary([1, 2, 3, 4, 5, 6, 7, 8], MODEL_NAME, EPOCH)
