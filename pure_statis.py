#!/usr/bin/python
import csv
import numpy as np
from six.moves import cPickle as pickle

# read in data
dataset = 'dataset/'
train_season_statis_filename = dataset + 'final_train_season_statis.pickle'
train_month_statis_filename = dataset + 'final_train_month_statis.pickle'
train_all_data_filename = dataset + 'final_train_data_statis.pickle'

test_season_statis_filename = dataset + 'final_test_season_statis.pickle'
test_month_statis_filename = dataset + 'final_test_month_statis.pickle'
test_all_data_filename = dataset + 'final_test_data_statis.pickle'

with open(train_season_statis_filename, 'rb') as f:
  train_season_statis = pickle.load(f)
with open(train_month_statis_filename, 'rb') as f:
  train_month_statis = pickle.load(f)
with open(train_all_data_filename, 'rb') as f:
  train_all_data = pickle.load(f)
with open(test_season_statis_filename, 'rb') as f:
  test_season_statis = pickle.load(f)
with open(test_month_statis_filename, 'rb') as f:
  test_month_statis = pickle.load(f)
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

train_data = np.column_stack((train_all_data[:,3105:], train_month_statis, train_season_statis))
test_data = np.column_stack((test_all_data[:,3105:], test_month_statis, test_season_statis))

print ('final, train data: ', train_data.shape)
print ('test data: ', test_data.shape)

print ('write data now..')

with open(dataset + 'final_train_all_statis.pickle', 'wb') as f:
  pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)
with open(dataset + 'final_test_all_statis.pickle', 'wb') as f:
  pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)
