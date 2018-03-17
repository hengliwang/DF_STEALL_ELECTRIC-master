import csv
import numpy as np
from six.moves import cPickle as pickle
import pandas as pd

dataset = '/media/dat1/liao/dataset/liao_statis/'
train_data_filename = dataset + 'train_data_norm.pickle'
test_data_filename = dataset + 'test_data_norm.pickle'


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
  _mean = np.nanmean(data, axis=1)
  _std = np.nanstd(data, axis=1)
  _var = np.nanvar(data, axis=1)
  _min = np.nanmin(data, axis=1)
  _max = np.nanmax(data, axis=1)
  _sum = np.nansum(data, axis=1)
  stat = np.column_stack((_mean, _std, _var, _min, _max, _sum))
  return stat


def get_statis_pd(data):
  df = pd.DataFrame(data)
  _mean = df.mean(axis=1,skipna=True).values
  _std = df.std(axis=1,skipna=True).values
  _var = df.var(axis=1,skipna=True).values
  _mad = df.mad(axis=1,skipna=True).values
  _median = df.median(axis=1,skipna=True).values
  _sum = df.sum(axis=1,skipna=True).values
  _skew = df.skew(axis=1,skipna=True).values
  _kurt = df.kurt(axis=1,skipna=True).values
  _cumsum = df.cumsum(axis=1,skipna=True).values
  _cumprod = df.cumprod(axis=1,skipna=True).values
  _diff = df.diff(axis=1).values[:,1:]
  _pct_change = df.pct_change(axis=1).values[:,1:]
  stat = np.column_stack((_mean, _std, _var, _mad, _median, _sum, _skew, _kurt))
  return stat


train_statis = get_statis(tmp_train_data)
test_statis = get_statis(tmp_test_data)
#train_statis = get_statis_pd(tmp_train_data)
#test_statis = get_statis_pd(tmp_test_data)
train_data = np.column_stack((train_data, train_statis))
test_data = np.column_stack((test_data, test_statis))

with open(dataset + 'train_data_norm_statis.csv', 'wb') as f:
  writer = csv.writer(f, quoting=csv.QUOTE_NONE)
  for line in train_data:
    writer.writerow(line)
with open(dataset + 'test_data_norm_statis.csv', 'wb') as f:
  writer = csv.writer(f, quoting=csv.QUOTE_NONE)
  for line in test_data:
    writer.writerow(line)

with open(dataset + 'train_data_norm_statis.pickle', 'wb') as f:
  pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)
  
with open(dataset + 'test_data_norm_statis.pickle', 'wb') as f:
  pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)
