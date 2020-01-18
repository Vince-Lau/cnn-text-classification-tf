# -*- coding:utf-8 -*-
# @Author  : Yuanjing Liu
# @Desc    : 
# @Time    : 2020/1/17 19:36

from clean_str import *
import pandas as pd
from sklearn.model_selection import train_test_split


class SampleSplit():
    def __init__(self, clean_pth):
        self.SEED = 10
        self.CLS = CleanStr()
        self.CLEAN_DATA_PTH = clean_pth

    def shuffle_sample(self, sample, sample_per):
        np.random.seed(self.SEED)
        shuffle_indices = np.random.permutation(np.arange(len(sample)))
        shuffle_sample = sample[shuffle_indices]
        dev_sample_index = -1 * int(sample_per * float(len(sample)))
        sample_train, sample_test = shuffle_sample[:dev_sample_index], shuffle_sample[dev_sample_index:]
        return sample_train, sample_test

    def split_dataset_(self, id_text):
        # 归类
        print('训练集与测试集划分---------')
        real_label_list = list(map(lambda x: x.lower(), self.LABELS))
        train_dataset, test_dataset = [], []
        counti = 0
        for l1 in real_label_list:
            labels_list = []
            for indxi, line in enumerate(id_text):
                labels = line[-1].strip()
                if labels == l1:
                    labels_list.append(line)
                elif labels not in real_label_list:
                    counti += 1
                    print('第%d条记录错误, 共有: %d ----' % (indxi, counti))
                    print('内容为：{}'.format(line))
            labels_train, labels_test = self.shuffle_sample(labels_list)
            print("类别%s的train/test为：%d/%d, 相等：%s" % (l1, len(labels_train), len(labels_test),
                                                     (len(labels_list) == len(labels_train)+len(labels_test))))
            train_dataset.append(labels_train)
            test_dataset.append(labels_test)
        return train_dataset, test_dataset

    def write_txt(self, split_text):
        counti = 0
        for indxi, line in enumerate(split_text):
            labels = line[-1].strip()
            real_label_list = list(map(lambda x: x.lower(), self.LABELS))
            if labels in real_label_list:
                # 归类
                label_name = '_'.join(labels.split(' '))
                label_pth = os.path.join(self.INPUT_DATA_PTH, label_name)
                txt_pth = os.path.join(label_pth, '_'.join([label_name, str(indxi)]))

                # 内容提取
                con_text = " ".join(line[:-1])
                con_text = re.sub(r"\s{2,}", " ", con_text)
                con_text = con_text.strip()

                # 写入文件
                with open(txt_pth, 'w') as f:
                    f.write(con_text)

            else:
                counti += 1
                print('第%d条记录错误，共有%d ----' % (indxi, counti))
                print('内容为：{}'.format(line))

        return 0

    def split_sample_a1(self, split_rate=0.2, rand_seed=10):
        '''
        各类别均匀划分测试集与训练集
        '''
        labels_target = ['make up', 'skin care', 'bodycare', 'fashion']
        id_pd = pd.read_csv(self.CLEAN_DATA_PTH)
        labels_count = id_pd.groupby('labels').count()
        labels_low = list(labels_count[labels_count['content'] < 500].index)
        id_pd['labels'].replace(list(id_pd['labels'][~id_pd['labels'].isin(labels_target)]), 'others', inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(id_pd['content'], id_pd['labels'], test_size=split_rate,
                                                            stratify=id_pd['labels'], random_state=rand_seed)

        print('数据集划分完成')
        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        print('----------')
        print(pd.value_counts(y_train))
        print('----------')
        print(pd.value_counts(y_test))
        return [X_train, X_test, y_train, y_test]

    def split_sample_a2(self, split_rate=0.2, rand_seed=10):
        '''
        对大类别欠采样，小类别不变
        '''
        labels_target = ['make up', 'skin care', 'bodycare', 'fashion']
        id_pd = pd.read_csv(self.CLEAN_DATA_PTH)
        id_pd['labels'].replace(list(id_pd['labels'][~id_pd['labels'].isin(labels_target)]), 'others', inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(id_pd['content'], id_pd['labels'], test_size=split_rate,
                                                            stratify=id_pd['labels'], random_state=rand_seed)
        _, make_X_train, _, make_y_train = train_test_split(
            id_pd.loc[X_train.index, :][id_pd['labels'] == 'make up']['content'],
            id_pd.loc[X_train.index, :][id_pd['labels'] == 'make up']['labels'],
            test_size=0.4,
            stratify=id_pd.loc[X_train.index, :][id_pd['labels'] == 'make up']['labels'],
            random_state=10)

        _, skin_X_train, _, skin_y_train = train_test_split(
            id_pd.loc[X_train.index, :][id_pd['labels'] == 'skin care']['content'],
            id_pd.loc[X_train.index, :][id_pd['labels'] == 'skin care']['labels'],
            test_size=0.5,
            stratify=id_pd.loc[X_train.index, :][id_pd['labels'] == 'skin care']['labels'],
            random_state=10)

        new_X_train = np.concatenate((list(make_X_train),
                                      list(skin_X_train),
                                      list(id_pd.loc[X_train.index, :][id_pd['labels'] == 'fashion']['content']),
                                      list(id_pd.loc[X_train.index, :][id_pd['labels'] == 'others']['content']),
                                      list(id_pd.loc[X_train.index, :][id_pd['labels'] == 'bodycare']['content'])
                                      ), axis=0)
        new_y_train = np.concatenate((list(make_y_train),
                                      list(skin_y_train),
                                      list(id_pd.loc[X_train.index, :][id_pd['labels'] == 'fashion']['labels']),
                                      list(id_pd.loc[X_train.index, :][id_pd['labels'] == 'others']['labels']),
                                      list(id_pd.loc[X_train.index, :][id_pd['labels'] == 'bodycare']['labels'])
                                      ), axis=0)

        print('数据集划分完成')
        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        print('----------')
        print(pd.value_counts(y_train))
        print('----------')
        print(pd.value_counts(y_test))
        return [new_X_train, X_test, new_y_train, y_test]

    def split_sample_a3(self, ori_data_path, split_rate=0.2, rand_seed=10):
        '''
        对大类别欠采样，小类别过采样
        '''
        pass


if __name__ == '__main__':
    Sp = SampleSplit()