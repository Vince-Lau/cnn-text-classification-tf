
It is slightly simplified implementation of Kim's [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) paper in Tensorflow.

## 样本准备
1. 使用`datawork`进行样本数据整理，脚本参考`fmd_dw_zxx`空间中的研发测试下`/liuyu/lyj_get_sample_for_classify`，将样本数据保存在`t_load_text_classify_sample`表格。
2. 使用odpscmd命令行工具，将数据下载到开发环境，命令如下
```bash
tunnel download -fd='||' -rd='|-|' t_load_text_classify_sample text_classify_sample.txt;
```
3. 在本地使用脚本`./nlp_bash/clean_str.py`清洗数据，并保存
4. 上传原始样本数据和清理后的样本数据到oss上备份，测试样本放在`oss://devops-fimedia-bucket/fm-algo-machine-classify/data/`,
确定使用数据集训练生产模型，放在生产路径`oss://algo-fimedia-bucket/algo-machine-classify/data/`，
如`id_ClassII_20200222_all.txt`和`id_ClassII_20200222_all_clean.txt`

## Requirements

- Python 3
- Tensorflow > 0.12
- Numpy

## Training

Print parameters:

```bash
./train.py --help
```

```
optional arguments:
  -h, --help            show this help message and exit
  --embedding_dim EMBEDDING_DIM
                        Dimensionality of character embedding (default: 128)
  --enable_word_embeddings
                        Enable/disable the word embeddings (default: True)
  --filter_sizes FILTER_SIZES
                        Comma-separated filter sizes (default: '3,4,5')
  --num_filters NUM_FILTERS
                        Number of filters per filter size (default: 128)
  --l2_reg_lambda L2_REG_LAMBDA
                        L2 regularizaion lambda (default: 0.0)
  --dropout_keep_prob DROPOUT_KEEP_PROB
                        Dropout keep probability (default: 0.5)
  --batch_size BATCH_SIZE
                        Batch Size (default: 64)
  --num_epochs NUM_EPOCHS
                        Number of training epochs (default: 100)
  --evaluate_every EVALUATE_EVERY
                        Evaluate model on dev set after this many steps
                        (default: 100)
  --checkpoint_every CHECKPOINT_EVERY
                        Save model after this many steps (default: 100)
  --allow_soft_placement ALLOW_SOFT_PLACEMENT
                        Allow device soft device placement
  --noallow_soft_placement
  --log_device_placement LOG_DEVICE_PLACEMENT
                        Log placement of ops on devices
  --nolog_device_placement

```

Train:

```bash
./train.py
```

## Evaluating

```bash
./eval.py --eval_train --checkpoint_dir="./runs/1459637919/checkpoints/"
```

Replace the checkpoint dir with the output from the training. To use your own data, change the `eval.py` script to load your data.


## References

- **[This code belongs to the "Implementing a CNN for Text Classification in Tensorflow" blog post.](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)**
- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)
