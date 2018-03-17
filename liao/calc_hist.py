import csv
import numpy as np
from six.moves import cPickle as pickle
import pandas as pd

# read in data
dataset = '/media/dat1/liao/dataset/liao_statis/'
train_data_filename = dataset + 'train_data_norm_statis.pickle'
train_label_filename = dataset + 'train_label.pickle'
test_data_filename = dataset + 'test_data_norm_statis.pickle'
test_uid_filename = dataset + 'test_uid.pickle'

with open(train_data_filename, 'rb') as f:
  train_data = pickle.load(f)
with open(train_label_filename, 'rb') as f:
  train_label = pickle.load(f)
with open(test_data_filename, 'rb') as f:
  test_data = pickle.load(f)
with open(test_uid_filename, 'rb') as f:
  test_uid = pickle.load(f)


def get_hist(data):
  hist = list()
  for line in data:
    h, _ = np.histogram(line, bins=[-1] + np.arange(0,1100,100).tolist())
    hist.append(h)
  print(np.array(hist).shape)
  return np.array(hist)

train_data = np.column_stack((train_data, get_hist(train_data[:,:365])))
test_data = np.column_stack((test_data, get_hist(test_data[:,:365])))

with open(dataset + 'train_data_hist_statis.csv', 'wb') as f:
  writer = csv.writer(f, quoting=csv.QUOTE_NONE)
  for line in train_data:
    writer.writerow(line)
with open(dataset + 'test_data_hist_statis.csv', 'wb') as f:
  writer = csv.writer(f, quoting=csv.QUOTE_NONE)
  for line in test_data:
    writer.writerow(line)

with open(dataset + 'train_data_hist_statis.pickle', 'wb') as f:
  pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)
  
with open(dataset + 'test_data_hist_statis.pickle', 'wb') as f:
  pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)
  
