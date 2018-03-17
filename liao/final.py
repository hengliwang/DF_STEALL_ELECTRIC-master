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

f_data = open(pickle_file_data)
train_data= pickle.load(f_data)
f_label = open(pickle_file_label)
train_label = pickle.load(f_label)
f_label.close()
f_data.close()
######
pickle_file_data = dataset + 'test_data_statis.pickle'
f_data = open(pickle_file_data)
test_data = pickle.load(f_data)
f_data.close()
######
pickle_file_data = dataset + 'test_uid.pickle'
f_data = open(pickle_file_data)
test_uid = pickle.load(f_data)
f_data.close()


from sklearn.decomposition import PCA

#train_data_refine = use_statis(train_data)
#with open(dataset + 'train_data_refine_statis.pickle', 'wb') as f:
#  pickle.dump(train_data_refine, f, pickle.HIGHEST_PROTOCOL)

print(train_data.shape)
print(test_data.shape)
pca = PCA(60).fit(train_data)
train_data = pca.transform(train_data)
test_data = pca.transform(test_data)
#train_data = train_data[:,-11:]
#test_data = test_data[:,-11:]


from sklearn import linear_model, neighbors, svm, tree, preprocessing
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

print ("Transform OK!")
print ("Let's train the model")
print("train_data",train_data.shape)
print("train_label",train_label.shape)

#trian_data = preprocessing.normalize(train_data, norm='l2')
#test_data = preprocessing.normalize(test_data, norm='l2')


#-----------------------------------------------------------------------------------
#clf1 = linear_model.LogisticRegression(random_state=1)
clf = GradientBoostingClassifier(n_estimators=2500, #learning_rate=1.0,
       verbose = 1, random_state=1).fit(train_data, train_label)
#clf3 = GaussianNB()
#eclf1 = VotingClassifier(estimators=[('lr', clf1), ('gb', clf2), ('gnb', clf3)], voting='hard')
#eclf1 = eclf1.fit(train_data, train_label)
#result = eclf1.predict_prob()
#-----------------------------------------------------------------------------------

#clf = AdaBoostClassifier(n_estimators=3000)
clf.fit(train_data, train_label)

#clf = BaggingClassifier(n_estimators = 2000)
#clf.fit(train_data, train_label)


#eclf2 = VotingClassifier(estimators=[('lr', clf1), ('gb', clf2), ('gnb', clf3)], voting='soft')

#clf  = RandomForestClassifier(n_estimators=6000, max_depth = 4, verbose=1).fit(train_data, train_label)
#knn = neighbors.KNeighborsClassifier()
#logistic = linear_model.LogisticRegression()
#clf = svm.SVC(probability = True)
#clf = tree.DecisionTreeClassifier()


#print('KNN score: %f' % knn.fit(train_data, train_label).score(valid_data, valid_label))
#result = knn.fit(train_data, train_label).predict_proba(test_data)
#train_data = train_data[0:5000,:]
#train_label = train_label[0:5000]
#result = logistic.fit(train_data, train_label).predict_proba(test_data)
#clf.fit(train_data, train_label)
result = clf.predict_proba(test_data)
#result = clf.predict_proba(test_data)

class Score:
  def __init__(self, uid, prob):
    self.uid = uid
    self.prob = prob

print result

scores = list()
for i in range(len(test_uid)):
  scores.append(Score(test_uid[i], result[i][1]))

scores = sorted(scores, key=lambda x: x.prob, reverse=True)


ff = open('raw.csv', 'w')
for i in range(len(test_uid)):
    uid = scores[i].uid
    ff.write(uid + ',' + str(scores[i].prob) + '\n')
ff.close()


with open('result.csv', 'w') as f:
  for i in range(len(test_uid)):
    f.write(scores[i].uid + '\n')

