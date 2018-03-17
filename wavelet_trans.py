#!/usr/bin/python
import pywt 
import numpy as np
from six.moves import cPickle as pickle

dataset = 'dataset/'
train_data_filename = dataset + 'final_train_data.pickle'
test_data_filename = dataset + 'final_test_data.pickle'

with open(train_data_filename, 'rb') as f:
  train_data = pickle.load(f)

with open(test_data_filename, 'rb') as f:
  test_data = pickle.load(f)

train_cA, train_cD = pywt.dwt(train_data, 'db2')
test_cA, test_cD = pywt.dwt(test_data, 'db2')

print (train_cA.shape, train_cD.shape)
print (test_cA.shape, test_cD.shape)

train_dwt = np.column_stack((train_cA, train_cD))
test_dwt = np.column_stack((test_cA, test_cD))

with open(dataset + 'final_train_dwt.pickle', 'wb') as f:
  pickle.dump(train_dwt, f, pickle.HIGHEST_PROTOCOL)
with open(dataset + 'final_test_dwt.pickle', 'wb') as f:
  pickle.dump(test_dwt, f, pickle.HIGHEST_PROTOCOL)


