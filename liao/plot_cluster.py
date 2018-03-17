import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import os
from sklearn import cluster, datasets
from six.moves import cPickle as pickle
import numpy as np
from sklearn import linear_model, neighbors, svm, tree, preprocessing

# load data
dataset = '/media/dat1/liao/dataset/'
#train_data_filename = dataset + '/dct_dataset/train_data.pickle'
train_data_filename = dataset + 'train_delta_data.pickle'
train_label_filename = dataset + 'train_delta_label.pickle'

#test_data_filename = dataset + 'test_data.pickle'
#test_uid_filename = dataset + 'test_uid.pickle'

with open(train_data_filename) as f:
  train_data= pickle.load(f)
with open(train_label_filename) as f:
  train_label = pickle.load(f)
#with open(test_data_filename) as f:
#  test_data = pickle.load(f)
#with open(test_uid_filename) as f:
#  data_id = pickle.load(f)

# do PCA
#pca = PCA(200).fit(train_data)
#train_data = pca.transform(train_data)
#test_data = pca.transform(test_data)

#train_data = scale(train_data)
trian_data = preprocessing.normalize(train_data, norm='l2')

n_samples_train, n_features_train = train_data.shape
#n_samples_test, n_features_test = test_data.shape

for n_cluster in range(2,10):

	print("n_cluster: %d, \t n_samples %d, \t n_features %d"
	      % (n_cluster, n_samples_train, n_features_train))


	#reduced_data = PCA(n_components=2).fit_transform(train_data)
	clf = KMeans(init='k-means++', n_clusters=n_cluster, n_init=20)
	clf.fit(train_data)
	#clf = cluster.SpectralClustering(n_clusters=2).fit(train_data)


	predict_label = clf.labels_


	def func(pre_label):
	  n = 0
	  for i in range(n_samples_train):
	    if predict_label[i] == pre_label and train_label[i] == 1:n+=1
	  return n


	for i in range(n_cluster):
	  print (str(sum(predict_label==i)) + ' ' + str(func(i)))

