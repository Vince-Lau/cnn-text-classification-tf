# -*- coding:utf-8 -*-
# @Author  : Yuanjing Liu
# @Desc    : 
# @Time    : 2020/2/18 17:27

from nlp_base.clean_str import *
import pandas as pd
import yaml
# import os, sys
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler


with open("../config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)


class SampleSplitII():
    def __init__(self):
        self.rand_seed = cfg["data_processing"]["sample_seed"]
        self.labels_other = cfg["data_processing"]["labels_other"]
        self.ori_path = cfg["data_processing"]["ori_path"]
        ori_name = cfg["data_processing"]["current_data"]
        clean_name = "_".join([ori_name.split(".")[0], "clean"]) + '.csv'
        self.clean_text_path = os.path.join(self.ori_path, clean_name)
        # label2id
        label_path = cfg["data_processing"]["label_path"]
        self.label2id = pd.read_csv(label_path)
        self.labels_all = self.label2id["labels"].values
        self.labels_target = list(filter(lambda x: x != self.labels_other, self.labels_all))
        self.labels_name_num = dict(self.label2id[["category_name", "labels"]].values)

    def data_deal_other(self):
        data = pd.read_csv(self.clean_text_path)
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
        num_to_name = {v: k for k, v in self.labels_name_num.items()}
        labels_name = [num_to_name[ni] for ni in labels_class]
        return np.array(labels_vec), labels_name

    def split_sample_random(self, split_rate=0.2):
        data = self.data_deal_other()
        data_not_other = data[data['labels'] != self.labels_other]
        data_other = data[data['labels'] == self.labels_other].sample(n=5000, random_state=self.rand_seed)
        X_train, X_test, y_train, y_test = train_test_split(data_not_other['content'],
                                                            data_not_other['labels'],
                                                            test_size=split_rate,
                                                            stratify=data_not_other['labels'],
                                                            random_state=self.rand_seed)
        # other data split
        _X_train, _X_test, _y_train, _y_test = train_test_split(data_other['content'],
                                                           data_other['labels'],
                                                           test_size=split_rate,
                                                           stratify=data_other['labels'],
                                                           random_state=self.rand_seed)

        # 拼接组合新数据
        X_train, X_test, y_train, y_test = np.concatenate((X_train.values, _X_train.values)), \
                                           np.concatenate((X_test.values, _X_test.values)), \
                                           np.concatenate((y_train.values, _y_train.values)), \
                                           np.concatenate((y_test.values, _y_test.values))

        dataset = self.data_deal_dataset(X_train, X_test, y_train, y_test)
        return dataset

    def split_sample_num(self, split_rate=0.2):
        '''
        对多样本进行欠采样
        :param data: datafram, text data
        :param split_rate: float, split rate
        :param rand_seed: int, rand seed
        :return:
        '''

        label_few = [53, 57, 102, 48, 43, 66, 14, 19, 46, 21, 81, 60, 49, 74]
        label_most = [112, 64, 34]
        sample_num = 3000

        data = self.data_deal_other()
        data_new = data[data['labels'].isin(label_few)]
        for di in label_most:
            data_new = pd.concat([data_new, data[data.labels == di].sample(n=sample_num, random_state=self.rand_seed)])

        X_train, X_test, y_train, y_test = train_test_split(data_new['content'],
                                                            data_new['labels'],
                                                            test_size=split_rate,
                                                            stratify=data_new['labels'],
                                                            random_state=self.rand_seed)
        dataset = self.data_deal_dataset(X_train, X_test, y_train, y_test)
        return dataset

    def split_sample_over_random(self, split_rate=0.2):
        sample_num = 3000
        label_most = [49, 74, 112, 64, 34]

        data = self.data_deal_other()
        data_new = data[~data['labels'].isin(label_most)]
        for di in label_most:
            data_new = pd.concat([data_new, data[data.labels == di].sample(n=sample_num, random_state=self.rand_seed)])
        data_new = data_new.reset_index(drop=True)

        # 对少数样本过采样
        ros = RandomOverSampler(sampling_strategy='not majority', random_state=self.rand_seed)
        feature = np.concatenate((np.array([data_new.index]).T, data_new.values), axis=1)
        X_resampled, y_resampled = ros.fit_sample(feature, data_new['labels'].values)
        X_train, X_test, y_train, y_test = train_test_split(X_resampled.T[1],
                                                            y_resampled,
                                                            test_size=split_rate,
                                                            stratify=y_resampled,
                                                            random_state=self.rand_seed)
        dataset = self.data_deal_dataset(X_train, X_test, y_train, y_test)
        return dataset

    def split_sample_over_random2(self, split_rate=0.2, rand_seed=10):
        sample_num = 4000
        label_most = [112, 64, 34]

        # 过滤太多的数据，重新组合数据
        data = self.data_deal_other()
        data_new = data[~data['labels'].isin(label_most)]

        for di in label_most:
            data_new = pd.concat([data_new, data[data.labels == di].sample(n=sample_num, random_state=rand_seed)])
        data_new = data_new.reset_index(drop=True)

        # 划出测试集
        X_train, X_test, y_train, y_test = train_test_split(data_new['content'],
                                                            data_new['labels'],
                                                            test_size=split_rate,
                                                            stratify=data_new['labels'],
                                                            random_state=rand_seed)
        # 对少数样本过采样
        ros = RandomOverSampler(sampling_strategy='not majority', random_state=rand_seed)
        X_train = np.concatenate((np.array([X_train.index]).T, np.array([X_train.values]).T), axis=1)
        X_train, y_train = ros.fit_sample(X_train, y_train.values)
        dataset = self.data_deal_dataset(X_train.T[1], X_test, y_train, y_test)
        return dataset


if __name__ == '__main__':
    Sp = SampleSplitII()
    # dataset_a1 = Sp.split_sample_random()
    dataset_a2 = Sp.split_sample_over_random2()
