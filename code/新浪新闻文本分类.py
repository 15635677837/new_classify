# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:39:10 2019

@author: 科
"""
# 4.1 获取文本文件路径
import os

def getFilePathList(rootDirPath):
    filePath_list = []
    for walk in os.walk(rootDirPath):
        part_filePath_list = [os.path.join(walk[0], file) for file in walk[2]]
        filePath_list.extend(part_filePath_list)
    return filePath_list
filePath_list = getFilePathList('THUCNews')
len(filePath_list)

# 4.2 获取所有样本标签
label_list = []
for filePath in filePath_list:
    label = filePath.split('\\')[1]
    label_list.append(label)
len(label_list)

# 4.3 标签统计计数
import pandas as pd

pd.value_counts(label_list)

# 4.4 调用pickle库保存label_list
import pickle

with open('label_list.pickle', 'wb') as file:
    pickle.dump(label_list, file)
    
# 4.5 获取所有样本内容、保存content_list
import time
import pickle
import re

def getFile(filePath):
    with open(filePath, encoding='utf8') as file:
        fileStr = ''.join(file.readlines(1000))
    return fileStr

interval = 20000
n_samples = len(label_list)
startTime = time.time()
directory_name = 'content_list'
if not os.path.isdir(directory_name):
    os.mkdir(directory_name)
for i in range(0, n_samples, interval):
    startIndex = i
    endIndex = i + interval
    content_list = []
    print('%06d-%06d start' %(startIndex, endIndex))
    for filePath in filePath_list[startIndex:endIndex]:
        fileStr = getFile(filePath)
        content = re.sub('\s+', ' ', fileStr)
        content_list.append(content)
    save_fileName = directory_name + '/%06d-%06d.pickle' %(startIndex, endIndex)
    with open(save_fileName, 'wb') as file:
        pickle.dump(content_list, file)
    used_time = time.time() - startTime
    print('%06d-%06d used time: %.2f seconds' %(startIndex, endIndex, used_time))
# 5.加载数据
import time
import pickle
import os

def getFilePathList(rootDir):
    filePath_list = []
    for walk in os.walk(rootDir):
        part_filePath_list = [os.path.join(walk[0], file) for file in walk[2]]
        filePath_list.extend(part_filePath_list)
    return filePath_list

startTime = time.time()
contentListPath_list = getFilePathList('content_list')
content_list = []
for filePath in contentListPath_list:
    with open(filePath, 'rb') as file:
        part_content_list = pickle.load(file)
    content_list.extend(part_content_list)
with open('label_list.pickle', 'rb') as file:
    label_list = pickle.load(file)
used_time = time.time() - startTime
print('used time: %.2f seconds' %used_time)
sample_size = len(content_list)
print('length of content_list，also called sample size: %d' %sample_size)
    
    
# 6.1 制作词汇表
from collections import Counter 
def getVocabularyList(content_list, vocabulary_size):
    allContent_str = ''.join(content_list)
    counter = Counter(allContent_str)
    vocabulary_list = [k[0] for k in counter.most_common(vocabulary_size)]
    return ['PAD'] + vocabulary_list
startTime = time.time()
vocabulary_list = getVocabularyList(content_list, 10000)
used_time = time.time() - startTime
print('used time: %.2f seconds' %used_time)

# 6.2 保存词汇表
import pickle 

with open('vocabulary_list.pickle', 'wb') as file:
    pickle.dump(vocabulary_list, file)
    
# 6.3 加载词汇表
import pickle

with open('vocabulary_list.pickle', 'rb') as file:
    vocabulary_list = pickle.load(file)
# 7.数据准备
import time
startTime = time.time()
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(content_list, label_list)
train_content_list = train_X
train_label_list = train_y
test_content_list = test_X
test_label_list = test_y
used_time = time.time() - startTime
print('train_test_split used time : %.2f seconds' %used_time)
vocabulary_size = 10000  # 词汇表达小
sequence_length = 600  # 序列长度
embedding_size = 64  # 词向量维度
num_filters = 256  # 卷积核数目
filter_size = 5  # 卷积核尺寸
num_fc_units = 128  # 全连接层神经元
dropout_keep_probability = 0.5  # dropout保留比例
learning_rate = 1e-3  # 学习率
batch_size = 64  # 每批训练大小
word2id_dict = dict([(b, a) for a, b in enumerate(vocabulary_list)])
content2idList = lambda content : [word2id_dict[word] for word in content if word in word2id_dict]
train_idList_list = [content2idList(content) for content in train_content_list]
used_time = time.time() - startTime
print('content2idList used time : %.2f seconds' %used_time)
import numpy as np
num_classes = np.unique(label_list).shape[0]
import tensorflow.contrib.keras as kr
train_X = kr.preprocessing.sequence.pad_sequences(train_idList_list, sequence_length)
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
train_y = labelEncoder.fit_transform(train_label_list)
train_Y = kr.utils.to_categorical(train_y, num_classes)
import tensorflow as tf
tf.reset_default_graph()
X_holder = tf.placeholder(tf.int32, [None, sequence_length])
Y_holder = tf.placeholder(tf.float32, [None, num_classes])
used_time = time.time() - startTime
print('data preparation used time : %.2f seconds' %used_time)  

#8.搭建神经网络
embedding = tf.get_variable('embedding', 
                            [vocabulary_size, embedding_size])
embedding_inputs = tf.nn.embedding_lookup(embedding,
                                          X_holder)
conv = tf.layers.conv1d(embedding_inputs,
                        num_filters,
                        filter_size)
max_pooling = tf.reduce_max(conv, 
                            [1])
full_connect = tf.layers.dense(max_pooling,
                               num_fc_units)
full_connect_dropout = tf.contrib.layers.dropout(full_connect, 
                                                 keep_prob=dropout_keep_probability)
full_connect_activate = tf.nn.relu(full_connect_dropout)
softmax_before = tf.layers.dense(full_connect_activate,
                                 num_classes)
predict_Y = tf.nn.softmax(softmax_before)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_holder,
                                                           logits=softmax_before)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)
isCorrect = tf.equal(tf.argmax(Y_holder, 1), tf.argmax(predict_Y, 1))
accuracy = tf.reduce_mean(tf.cast(isCorrect, tf.float32)) 
    
    
    