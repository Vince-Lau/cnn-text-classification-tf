# -*- coding:utf-8 -*-
# @Author  : Yuanjing Liu
# @Desc    : 
# @Time    : 2020/1/14 16:47

import numpy as np
import os
import re
import pandas as pd
import shutil
# from sklearn.preprocessing import LabelBinarizer
# from tensorflow.contrib import learn

'''
tunnel下载数据集
tunnel download -fd='||' -rd='|-|' t_load_text_classify_sample id_ClassII_20200222_all.txt;
'''


class CleanStr():

    def __init__(self):
        self.ROW_SPLIT = '|-|'
        self.COLUMN_SPLIT = '||'
        self.STOP_PATH = '../data/id_all_text/labels/stopwords-id.txt'
        self.LABEL_PATH = '../data/id_all_text/labels/labelList_classII.txt'
        self.SAVE_CSV_PATH = '../data/id_all_text/ori/id_ClassII_20200222_all_clean.csv'
        self.stopwords = list(map(lambda x: x.strip().lower(), list(open(self.STOP_PATH, "r", encoding='latin-1').readlines())))
        self.LABELS = list(map(lambda x: x.strip().lower(), list(open(self.LABEL_PATH, "r", encoding='latin-1').readlines())))

    def clean_str(self, string):
        string = re.sub(r"\r\n", ".", string)
        string = re.sub(r"[^A-Za-z0-9-||]", " ", string)
        string = re.sub(r"\|\|", " || ", string)
        string = re.sub(r"\(|\)|\[|\]|\{|\}|<|\>", "", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def clean_stopwords(self, clean_str):
        clean_stopwords = " ".join(w for w in clean_str.split() if w not in self.stopwords)
        return clean_stopwords

    def get_str_list(self, data_pth):
        with open(data_pth, 'rb') as f:
            id_text = f.read().decode('ISO-8859-1')
        id_text = self.clean_str(id_text)
        id_text = self.clean_stopwords(id_text)
        id_text = id_text.split(self.ROW_SPLIT)
        id_text = list(map(lambda x: x.split(self.COLUMN_SPLIT), id_text))
        id_text = [list(map(lambda x: x.strip(), id_list)) for id_list in id_text if len(id_list) > 0]
        return id_text

    def get_content_list(self, data_pth):
        content_label = []
        counti = 0
        id_text = self.get_str_list(data_pth)
        for indxi, line in enumerate(id_text):
            labels = line[-1].strip()
            if labels in self.LABELS:
                # 内容提取
                con_text = " ".join(line[:-1])
                con_text = re.sub(r"\s{2,}", " ", con_text)
                con_text = con_text.strip()
                content_label.append([con_text, labels])
            else:
                counti += 1
                print('第%d条记录错误，共有%d ----' % (indxi, counti))
                print('内容为：{}'.format(line))
        return content_label

    def get_content_pd(self, data_pth, savefile=False):
        content_list = self.get_content_list(data_pth)
        content_pd = pd.DataFrame(content_list, columns=['content', 'labels'])
        content_pd.dropna(axis=0, how='any', inplace=True)
        if savefile:
            content_pd.to_csv(path_or_buf=self.SAVE_CSV_PATH, sep=',', header=True, index=False)
        return content_pd


if __name__ == '__main__':
    cs = CleanStr()
    _ = cs.get_content_pd('../data/id_all_text/ori/id_ClassII_20200222_all.txt', savefile=True)

