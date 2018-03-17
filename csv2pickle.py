from six.moves import cPickle as pickle
import numpy as np
import csv

dataset = '/media/dat1/liao/dataset/dct_dataset/'

def csv2pickle(csvfilename, pickle_filename):
  my_matrix = np.loadtxt(open(csvfilename,"rb"),delimiter=",",skiprows=0)
  with open(pickle_filename, 'wb') as f:
    pickle.dump(my_matrix, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
  csv2pickle(dataset + 'train_data.csv', dataset +  'train_data.pickle')
  csv2pickle(dataset + 'test_data.csv', dataset +  'test_data.pickle')
