import pandas as pd
from unidecode import unidecode
from sklearn.model_selection import KFold, train_test_split

from data_utils import clean_data
import utils


def convert_to_ascii(df):
    new_df = []
    for essay in df:
        new_df.append(unidecode(essay))
    return new_df


def create_dataset(fold=True):
    '''Run this function once to create train,val,test files for K folds'''
    data_all = pd.read_csv('asap/training_set_rel3.fixed.tsv.zip',
                           sep='\t', encoding='latin1')
    data_all['essay'] = convert_to_ascii(data_all['essay'])
    data_all['essay'] = clean_data(data_all['essay'])

    for p in range(1, 9):
        data_prompt = data_all[data_all['essay_set']
                               == p].reset_index(drop=True)
        print(data_prompt.head())

        if fold:
            kf = KFold(n_splits=5, shuffle=True, random_state=420)
            n = 1
            for train_index, test_index in kf.split(data_prompt):
                # print("TRAIN:", train_index[:10], "TEST:", test_index[:10])
                val_index = test_index[:len(test_index)//2]
                test_index = test_index[len(test_index)//2:]
                print(len(train_index), len(val_index), len(test_index))

                fold_path = utils.mkpath('asap/fold_{}/'.format(n))
                data_prompt.loc[train_index].to_csv(
                    fold_path + 'prompt_{}_train.tsv'.format(p), sep='\t', index=False)
                data_prompt.loc[val_index].to_csv(
                    fold_path + 'prompt_{}_val.tsv'.format(p), sep='\t', index=False)
                data_prompt.loc[test_index].to_csv(
                    fold_path + 'prompt_{}_test.tsv'.format(p), sep='\t', index=False)
                n += 1
        else:
            train, test = train_test_split(
                data_prompt, test_size=0.2, random_state=420, shuffle=False)
            val = test[:len(test)//2]
            test = test[len(test)//2:]
            path = utils.mkpath('asap/')
            print(len(train), len(val), len(test))
            train.to_csv(path + 'prompt_{}_train.tsv'.format(p),
                         sep='\t', index=False)
            val.to_csv(path + 'prompt_{}_val.tsv'.format(p),
                       sep='\t', index=False)
            test.to_csv(path + 'prompt_{}_test.tsv'.format(p),
                        sep='\t', index=False)


if __name__ == "__main__":
    create_dataset(fold=True)
