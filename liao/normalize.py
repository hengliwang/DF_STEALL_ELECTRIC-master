import csv
import numpy as np
from six.moves import cPickle as pickle
import pandas as pd

# read in data
dataset = '/media/dat1/liao/dataset/liao_statis/'
train_data_filename = dataset + 'train_data.pickle'
train_label_filename = dataset + 'train_label.pickle'
test_data_filename = dataset + 'test_data.pickle'
test_uid_filename = dataset + 'test_uid.pickle'

with open(train_data_filename, 'rb') as f:
  train_data = pickle.load(f)
with open(train_label_filename, 'rb') as f:
  train_label = pickle.load(f)
with open(test_data_filename, 'rb') as f:
  test_data = pickle.load(f)
with open(test_uid_filename, 'rb') as f:
  test_uid = pickle.load(f)

def normalize(v):
  norm=np.linalg.norm(v)
  if norm==0: 
    return v
  return v*10000/norm

def do_norm(data):
  shape = data.shape
  data = np.reshape(data, -1)
  data = normalize(data)
  data = np.reshape(data, shape)
  return data


def solve_missing(data):
  for row in range(len(data)):
    for col in range(len(data[row])):
      if data[row][col] < 0.0:
        data[row][col] = -1
  return data


train_data = solve_missing(do_norm(train_data))
test_data = solve_missing(do_norm(test_data))


with open(dataset + 'train_data_norm.csv', 'wb') as f:
  writer = csv.writer(f, quoting=csv.QUOTE_NONE)
  for line in train_data:
    writer.writerow(line)
with open(dataset + 'test_data_norm.csv', 'wb') as f:
  writer = csv.writer(f, quoting=csv.QUOTE_NONE)
  for line in test_data:
    writer.writerow(line)

with open(dataset + 'train_data_norm.pickle', 'wb') as f:
  pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)
  
with open(dataset + 'test_data_norm.pickle', 'wb') as f:
  pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)


