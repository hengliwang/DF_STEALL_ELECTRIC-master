#!/usr/bin/python
import csv
import numpy as np
from six.moves import cPickle as pickle

dataset = 'dataset/'
train_data_filename = dataset + 'final_train_data.pickle'
test_data_filename = dataset + 'final_test_data.pickle'


with open(train_data_filename, 'rb') as f:
  train_data = pickle.load(f)
with open(test_data_filename, 'rb') as f:
  test_data = pickle.load(f)

tmp_train_data = np.array(train_data)
for i in range(len(tmp_train_data)):
  for j in range(len(tmp_train_data[i])):
    if tmp_train_data[i][j] < 0.0:
      tmp_train_data[i][j] = float('nan')

tmp_test_data = np.array(test_data)
for i in range(len(tmp_test_data)):
  suc = False
  for j in range(len(tmp_test_data[i])):
    if tmp_test_data[i][j] > 0.0:
      suc = True
  if not suc: continue
  for j in range(len(tmp_test_data[i])):
    if tmp_test_data[i][j] < 0.0:
      tmp_test_data[i][j] = float('nan')


def get_statis(data):
  _mean = np.nanmean(data, axis=1)#均值
  _std = np.nanstd(data, axis=1)#方差
  _var = np.nanvar(data, axis=1)#标准差
  _min = np.nanmin(data, axis=1)#最小值
  _max = np.nanmax(data, axis=1)#最大值
  _sum = np.nansum(data, axis=1)#和
  stat = np.column_stack((_mean, _std, _var, _min, _max, _sum))
  return stat


DAYS_PER_YEAR = 1035
#提取每年
train_e_statis = get_statis(tmp_train_data[:,:DAYS_PER_YEAR])
test_e_statis = get_statis(tmp_test_data[:,:DAYS_PER_YEAR])

train_s_statis = get_statis(tmp_train_data[:,DAYS_PER_YEAR:2*DAYS_PER_YEAR])
test_s_statis = get_statis(tmp_test_data[:,DAYS_PER_YEAR:2*DAYS_PER_YEAR])

train_d_statis = get_statis(tmp_train_data[:,2*DAYS_PER_YEAR:])
test_d_statis = get_statis(tmp_test_data[:,2*DAYS_PER_YEAR:])

train_data = np.column_stack((train_data, train_e_statis, train_s_statis, train_d_statis))
test_data = np.column_stack((test_data, test_e_statis, test_s_statis, test_d_statis))

print (train_data.shape)
print (test_data.shape)

print ('write pickle now...')
with open(dataset + 'final_train_data_statis.pickle', 'wb') as f:
  pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)
with open(dataset + 'final_test_data_statis.pickle', 'wb') as f:
  pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)
