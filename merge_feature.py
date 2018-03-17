#!/usr/bin/python
import csv
import numpy as np
from six.moves import cPickle as pickle

# read in data
dataset = 'dataset/'
train_season_statis_filename = dataset + 'final_train_season_statis.pickle'
train_month_statis_filename = dataset + 'final_train_month_statis.pickle'
train_dwt_filename = dataset + 'final_train_dwt.pickle'
train_all_data_filename = dataset + 'final_train_data_statis.pickle'

test_season_statis_filename = dataset + 'final_test_season_statis.pickle'
test_month_statis_filename = dataset + 'final_test_month_statis.pickle'
test_dwt_filename = dataset + 'final_test_dwt.pickle'
test_all_data_filename = dataset + 'final_test_data_statis.pickle'

with open(train_season_statis_filename, 'rb') as f:
  train_season_statis = pickle.load(f)
with open(train_month_statis_filename, 'rb') as f:
  train_month_statis = pickle.load(f)
with open(train_dwt_filename, 'rb') as f:
  train_dwt = pickle.load(f)
with open(train_all_data_filename, 'rb') as f:
  train_all_data = pickle.load(f)
with open(test_season_statis_filename, 'rb') as f:
  test_season_statis = pickle.load(f)
with open(test_month_statis_filename, 'rb') as f:
  test_month_statis = pickle.load(f)
with open(test_dwt_filename, 'rb') as f:
  test_dwt = pickle.load(f)
with open(test_all_data_filename, 'rb') as f:
  test_all_data = pickle.load(f)

from sklearn.decomposition import PCA

print ('train all data: ', train_all_data.shape)
print ('test all data: ', test_all_data.shape)

print ('construct train & test data...')

for i in range(len(train_all_data)):
  for j in range(len(train_all_data[i])):
    if train_all_data[i][j] != train_all_data[i][j]:
      train_all_data[i][j] = -1

train_data = PCA(100).fit_transform(train_all_data[:,:3105])
test_data = PCA(100).fit_transform(test_all_data[:,:3105])
train_data = np.column_stack((train_data, train_all_data[:,3105:]))
test_data = np.column_stack((test_data, test_all_data[:,3105:]))

print ('train month statis: ', train_month_statis.shape)
print ('test month statis: ', test_month_statis.shape)
print ('connect month statis...')
train_data = np.column_stack((train_data, PCA(50).fit_transform(train_month_statis)))
test_data = np.column_stack((test_data, PCA(50).fit_transform(test_month_statis)))

print ('train season statis: ', train_season_statis.shape)
print ('test season statis: ', test_season_statis.shape)
print ('connect season statis...')
train_data = np.column_stack((train_data, PCA(30).fit_transform(train_season_statis)))
test_data = np.column_stack((test_data, PCA(30).fit_transform(test_season_statis)))


print ('train dwt statis: ', train_dwt.shape)
print ('test dwt statis: ', test_dwt.shape)
print ('connect dwt...')
train_data = np.column_stack((train_data, PCA(5).fit_transform(train_dwt)))
test_data = np.column_stack((test_data, PCA(5).fit_transform(test_dwt)))


print ('final, train data: ', train_data.shape)
print ('test data: ', test_data.shape)

print ('write data now..')

with open(dataset + 'final_train_amsd.pickle', 'wb') as f:
  pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)
with open(dataset + 'final_test_amsd.pickle', 'wb') as f:
  pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)
