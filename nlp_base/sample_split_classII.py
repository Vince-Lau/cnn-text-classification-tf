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


class SampleSplitII():
    def __init__(self, clean_pth):
        self.SEED = 10
        self.labels_other = 34
        self.CLEAN_DATA_PTH = clean_pth
        self.labels_target = [64, 112, 74, 49, 60, 81, 21, 46, 19, 14, 66, 43, 48, 102, 57, 53]
        self.labels_all = [64, 112, 74, 49, 60, 81, 21, 46, 19, 14, 66, 43, 48, 102, 57, 53, 34]

    def data_deal_other(self):
        data = pd.read_csv(self.CLEAN_DATA_PTH)
        data.dropna(axis=0, how='any', inplace=True)
        data['labels'] = data['labels'].astype('int64')
        data['labels'].replace(list(data['labels'][~data['labels'].isin(self.labels_target)]), self.labels_other, inplace=True)
        return data

    def data_deal_dataset(self, X_train, X_test, y_train, y_test):
        print('数据集划分完成')
        print('训练集为: {}, 测试集为: {}'.format(X_train.shape[0], X_test.shape[0]))
        print('----------')
        print(pd.value_counts(y_train))
        print('----------')
        print(pd.value_counts(y_test))

        y_train, _ = self.labels_to_vector(list(y_train), self.labels_all)
        y_test, _ = self.labels_to_vector(list(y_test), self.labels_all)
        dataset = {'data': [X_train, X_test, y_train, y_test],
                   'labels_name': self.labels_all}
        return dataset

    def labels_to_vector(self, labels_list, labels_class=None):
        '''
        :param labels_list: 原始标签， list
        :param labels_class: 标签名字，list
        :return:
        '''
        labels_vec = []
        if set(labels_list) == set(labels_class):
            labels_class = np.array(labels_class)
            for li in labels_list:
                tmp_vec = np.zeros((len(labels_class)), dtype=int)
                tmp_vec[np.where(labels_class == li)] = 1
                labels_vec.append(list(tmp_vec))
        else:
            print('类别不对齐')
            print(' labels_list: %s\n labels_class: %s\n' % (set(labels_list), set(labels_class)))

            labels_class = np.array(set(labels_list))
            for li in labels_list:
                tmp_vec = np.zeros(labels_class.shape[1], dtype=int)
                tmp_vec[np.where(labels_class == li)] = 1
                labels_vec.append(list(tmp_vec))
        return np.array(labels_vec), np.array(labels_class)

    def split_sample_random(self, split_rate=0.2, rand_seed=10):
        '''
        分层抽样
        :param data:
        :param split_rate:
        :param rand_seed:
        :return:
        '''
        data = self.data_deal_other()
        data_not_other = data[data['labels'] != self.labels_other]
        data_other = data[data['labels'] == self.labels_other].sample(n=5000, random_state=rand_seed)
        X_train, X_test, y_train, y_test = train_test_split(data_not_other['content'],
                                                            data_not_other['labels'],
                                                            test_size=split_rate,
                                                            stratify=data_not_other['labels'],
                                                            random_state=rand_seed)
        # other data split
        _X_train, _X_test, _y_train, _y_test = train_test_split(data_other['content'],
                                                           data_other['labels'],
                                                           test_size=split_rate,
                                                           stratify=data_other['labels'],
                                                           random_state=rand_seed)

        # 拼接组合新数据
        X_train, X_test, y_train, y_test = np.concatenate((X_train.values, _X_train.values)), \
                                           np.concatenate((X_test.values, _X_test.values)), \
                                           np.concatenate((y_train.values, _y_train.values)), \
                                           np.concatenate((y_test.values, _y_test.values))

        dataset = self.data_deal_dataset(X_train, X_test, y_train, y_test)
        return dataset

    def split_sample_num(self, split_rate=0.2, rand_seed=10):
        '''
        对多样本进行过采样
        :param data: datafram, text data
        :param split_rate: float, split rate
        :param rand_seed: int, rand seed
        :return:
        '''

        label_few = [53,57,102,48,43,66,14,19,46,21,81,60,49,74]
        label_most = [112, 64, 34]
        sample_num = 3000

        data = self.data_deal_other()
        data_new = data[data['labels'].isin(label_few)]
        for di in label_most:
            data_new = pd.concat([data_new, data[data.labels == di].sample(n=sample_num, random_state=rand_seed)])

        X_train, X_test, y_train, y_test = train_test_split(data_new['content'],
                                                            data_new['labels'],
                                                            test_size=split_rate,
                                                            stratify=data_new['labels'],
                                                            random_state=rand_seed)
        dataset = self.data_deal_dataset(X_train, X_test, y_train, y_test)
        return dataset


if __name__ == '__main__':
    Sp = SampleSplitII('../data/id_all_text/ori/id_ClassII_20200222_all_clean.csv')
    # dataset_a1 = Sp.split_sample_random()
    dataset_a2 = Sp.split_sample_num()
