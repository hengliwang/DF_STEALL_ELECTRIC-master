import os
from six.moves import cPickle as pickle
import numpy as np
import csv

DAYS_OF_YEAR = 365
def use_statis(all_data):
  # start value
  start_mean_value = np.mean(all_data[:,:DAYS_OF_YEAR], axis=1)
  start_min_value = np.amin(all_data[:,:DAYS_OF_YEAR], axis=1)
  start_max_value = np.amax(all_data[:,:DAYS_OF_YEAR], axis=1)
  start_range_value = np.ptp(all_data[:,:DAYS_OF_YEAR], axis=1)
  start_quart_value = np.percentile(all_data[:,:DAYS_OF_YEAR], 25, axis=1)
  start_half_value = np.percentile(all_data[:,:DAYS_OF_YEAR], 50, axis=1)
  start_fif_value = np.percentile(all_data[:,:DAYS_OF_YEAR], 75, axis=1)
  start_median_value = np.median(all_data[:,:DAYS_OF_YEAR], axis=1)
  start_std_value = np.std(all_data[:,:DAYS_OF_YEAR], axis=1)
  start_var_value = np.var(all_data[:,:DAYS_OF_YEAR], axis=1)
#  start_special_value = start_mean_value / start_std_value
  # delta value
  delta_mean_value = np.mean(all_data[:,DAYS_OF_YEAR:], axis=1)
  delta_min_value = np.amin(all_data[:,DAYS_OF_YEAR:], axis=1)
  delta_max_value = np.amax(all_data[:,DAYS_OF_YEAR:], axis=1)
  delta_range_value = np.ptp(all_data[:,DAYS_OF_YEAR:], axis=1)
  delta_quart_value = np.percentile(all_data[:,DAYS_OF_YEAR:], 25, axis=1)
  delta_half_value = np.percentile(all_data[:,DAYS_OF_YEAR:], 50, axis=1)
  delta_fif_value = np.percentile(all_data[:,DAYS_OF_YEAR:], 75, axis=1)
  delta_median_value = np.median(all_data[:,DAYS_OF_YEAR:], axis=1)
  delta_std_value = np.std(all_data[:,DAYS_OF_YEAR:], axis=1)
  delta_var_value = np.var(all_data[:,DAYS_OF_YEAR:], axis=1)
#  delta_special_value = delta_mean_value / delta_std_value  
  # add to data
  all_data = np.column_stack((all_data, start_mean_value))
  all_data = np.column_stack((all_data, start_min_value))
  all_data = np.column_stack((all_data, start_max_value))
  all_data = np.column_stack((all_data, start_range_value))
  all_data = np.column_stack((all_data, start_quart_value))
  all_data = np.column_stack((all_data, start_half_value))
  all_data = np.column_stack((all_data, start_fif_value))
  all_data = np.column_stack((all_data, start_median_value))
  all_data = np.column_stack((all_data, start_std_value))
  all_data = np.column_stack((all_data, start_var_value))
#  all_data = np.column_stack((all_data, start_special_value))
  all_data = np.column_stack((all_data, delta_mean_value))
  all_data = np.column_stack((all_data, delta_min_value))
  all_data = np.column_stack((all_data, delta_max_value))
  all_data = np.column_stack((all_data, delta_range_value))
  all_data = np.column_stack((all_data, delta_quart_value))
  all_data = np.column_stack((all_data, delta_half_value))
  all_data = np.column_stack((all_data, delta_fif_value))
  all_data = np.column_stack((all_data, delta_median_value))
  all_data = np.column_stack((all_data, delta_std_value))
  all_data = np.column_stack((all_data, delta_var_value))
#  all_data = np.column_stack((all_data, delta_special_value))
  return all_data

#dataset = '/home/shaomingguang/shao_data/electric/dataset_1/'
dataset = '/media/dat1/liao/dataset/new_new_try/'
pickle_file_data = dataset + 'train_data_statis.pickle'
pickle_file_label = dataset + 'train_label.pickle'
test_filename = dataset + 'test_data_statis.pickle'

f_data = open(pickle_file_data)
train_data= pickle.load(f_data)
f_label = open(pickle_file_label)
train_label = pickle.load(f_label)
f_label.close()
f_data.close()
with open(test_filename) as f:
  test_data = pickle.load(f)


from sklearn.decomposition import PCA

#train_data = use_statis(train_data)i
train_data = train_data[:,-385:]
test_data = test_data[:,-385:]

with open(dataset + 'train_data_delta.pickle', 'wb') as f:
  pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)
  
with open(dataset + 'test_data_delta.pickle', 'wb') as f:
  pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)
 
#pca = PCA(50).fit(train_data)
#train_data = pca.transform(train_data)

##use_statis(train_data)
#
#from sklearn import linear_model, neighbors, svm, tree, preprocessing
#
#
#print ("Transform OK!")
#print ("Let's train the model")
#print("train_data",train_data.shape)
#print("train_label",train_label.shape)
#
##trian_data = preprocessing.normalize(train_data, norm='l2')
##test_data = preprocessing.normalize(test_data, norm='l2')
#
##knn = neighbors.KNeighborsClassifier()
#logistic = linear_model.LogisticRegression()
##clf = svm.SVC(probability = True)
##clf = tree.DecisionTreeClassifier()
#
#
##print('KNN score: %f' % knn.fit(train_data, train_label).score(valid_data, valid_label))
##result = knn.fit(train_data, train_label).predict_proba(test_data)
#new_train_data = train_data[0:4000,:]
#new_train_label = train_label[0:4000]
#result = logistic.fit(new_train_data, new_train_label).predict_proba(train_data)
##clf.fit(train_data, train_label)
##result = clf.predict_proba(test_data)
##result = clf.predict_proba(test_data)
#
#class Score:
#  def __init__(self, index, prob):
#    self.index = index
#    self.prob = prob
#
#print result
#
#scores = list()
#for i in range(len(train_data)):
#  scores.append(Score(i, result[i][1]))
#
#scores = sorted(scores, key=lambda x: x.prob, reverse=True)
#
#refine_train_data = list()
#refine_train_label = list()
#
#train_data = train_data.tolist()
#
#for i in range(len(train_data)):
#  if train_label[i] == 1:
#    refine_train_data.append(train_data[i])
#    refine_train_label.append(1)
#
#print (len(train_data))
#for i in range(len(train_data)-5000, len(train_data)):
#  refine_train_data.append(train_data[scores[i].index])
#  refine_train_label.append(0)
#
#refine_train_data_filename = dataset + 'train_data_refine.pickle'
#refine_train_label_filename = dataset + 'train_label_refine.pickle'
#
#with open(refine_train_data_filename, 'wb') as f:
#  pickle.dump(np.array(refine_train_data), f, pickle.HIGHEST_PROTOCOL)
#with open(refine_train_label_filename, 'wb') as f:
#  pickle.dump(np.array(refine_train_label), f, pickle.HIGHEST_PROTOCOL)
#
