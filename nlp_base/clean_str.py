# -*- coding:utf-8 -*-
# @Author  : Yuanjing Liu
# @Desc    : 
# @Time    : 2020/1/14 16:47

import numpy as np
import os
import yaml
import re
import pandas as pd

with open("../config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)


class CleanStr():

    def __init__(self):
        self.row_split = cfg["data_processing"]["row_split"]
        self.column_split = cfg["data_processing"]["column_split"]
        self.stop_word_path = cfg["data_processing"]["stop_word_path"]
        self.sys_label_path = cfg["data_processing"]["sys_label_path"]
        self.LABELS = list(map(lambda x: x.strip().lower(), list(open(self.sys_label_path, "r", encoding='latin-1').readlines())))

        self.ori_path = cfg["data_processing"]["ori_path"]
        ori_name = cfg["data_processing"]["current_data"]
        clean_name = "_".join([ori_name.split(".")[0], "clean"]) + '.csv'
        self.id_text_path = os.path.join(self.ori_path, ori_name)
        self.clean_text_path = os.path.join(self.ori_path, clean_name)

    def clean_str(self, string):
        string = re.sub(r"\r\n", ".", string)
        string = re.sub(r"[^A-Za-z0-9-||]", " ", string)
        string = re.sub(r"\|\|", " || ", string)
        string = re.sub(r"\(|\)|\[|\]|\{|\}|<|\>", "", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def clean_stopwords(self, clean_str):
        stopwords = list(
            map(lambda x: x.strip().lower(), list(open(self.stop_word_path, "r", encoding='latin-1').readlines())))
        clean_stopwords = " ".join(w for w in clean_str.split() if w not in stopwords)
        return clean_stopwords

    def get_str_list(self, data_pth):
        with open(data_pth, 'rb') as f:
            id_text = f.read().decode('ISO-8859-1')
        id_text = self.clean_str(id_text)
        id_text = self.clean_stopwords(id_text)
        id_text = id_text.split(self.row_split)
        id_text = list(map(lambda x: x.split(self.column_split), id_text))
        id_text = [list(map(lambda x: x.strip(), id_list)) for id_list in id_text if len(id_list) > 0]
        return id_text

    def get_content_list(self, data_pth):
        content_label = []
        counti = 0
        LABELS = list(
            map(lambda x: x.strip().lower(), list(open(self.sys_label_path, "r", encoding='latin-1').readlines())))
        id_text = self.get_str_list(data_pth)
        for indxi, line in enumerate(id_text):
            labels = line[-1].strip()
            if labels in LABELS:
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

    def get_content_pd(self, savefile=False):
        content_list = self.get_content_list(self.id_text_path)
        content_pd = pd.DataFrame(content_list, columns=['content', 'labels'])
        content_pd.dropna(axis=0, how='any', inplace=True)
        if savefile:
            content_pd.to_csv(path_or_buf=self.clean_text_path, sep=',', header=True, index=False)
        return content_pd


if __name__ == '__main__':
    cs = CleanStr()
    cs.get_content_pd(savefile=True)

