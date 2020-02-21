# -*- coding:utf-8 -*-
# @Author  : Yuanjing Liu
# @Desc    : 
# @Time    : 2020/2/18 17:27

# -*- coding:utf-8 -*-
# @Author  : Yuanjing Liu
# @Desc    :
# @Time    : 2020/1/17 19:36

from nlp_base.clean_str import *
import pandas as pd
import gc
from sklearn.model_selection import train_test_split


class SampleSplit():
    def __init__(self, clean_pth):
        self.SEED = 10
        self.CLS = CleanStr()
        self.CLEAN_DATA_PTH = clean_pth

    def split_sample_a1(self, split_rate=0.2, rand_seed=10):
        '''
        各类别均匀划分测试集与训练集
        '''
        labels_target = ['make up', 'skin care', 'bodycare', 'fashion']
        labels_cl = ['make up', 'skin care', 'bodycare', 'fashion', 'others']

        id_pd = pd.read_csv(self.CLEAN_DATA_PTH)
        id_pd.dropna(axis=0, how='any', inplace=True)
        id_pd['labels'].replace(list(id_pd['labels'][~id_pd['labels'].isin(labels_target)]), 'others', inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(id_pd['content'], id_pd['labels'], test_size=split_rate,
                                                            stratify=id_pd['labels'], random_state=rand_seed)

        print('数据集划分完成')
        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        print('----------')
        print(pd.value_counts(y_train))
        print('----------')
        print(pd.value_counts(y_test))

        y_train, labels_name = self.labels_to_vector(list(y_train), labels_cl)
        y_test, labels_name = self.labels_to_vector(list(y_test), labels_cl)
        dataset = {'data': [np.array(X_train), np.array(X_test), y_train, y_test],
                   'labels_name': labels_name}


if __name__ == '__main__':
    Sp = SampleSplit('./data/id_all_text/ori/id_ClassI_20200116_all_clean.csv')
    dataset_a1 = Sp.split_sample_a1()
    # dataset_a2 = Sp.split_sample_a2()
