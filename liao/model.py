import os
from six.moves import cPickle as pickle
import numpy as np
import csv

#dataset = '/home/shaomingguang/shao_data/electric/dataset_1/'
dataset = '/media/dat1/liao/dataset/new_try/'
train_data_filename = dataset + 'train_data_delta.pickle'
train_label_filename = dataset + 'train_label.pickle'
test_data_filename = dataset + 'test_data_delta.pickle'
test_uid_filename = dataset + 'test_uid.pickle'

with open(train_data_filename, 'rb') as f:
  train_data = pickle.load(f)
with open(train_label_filename, 'rb') as f:
  train_label = pickle.load(f)
with open(test_data_filename, 'rb') as f:
  test_data = pickle.load(f)
with open(test_uid_filename, 'rb') as f:
  test_uid = pickle.load(f)

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

print(train_data.shape)
print(test_data.shape)

# LDA
#lda = LinearDiscriminantAnalysis(n_components=50).fit(train_data, train_label)
#train_data = lda.transform(train_data)
#test_data = lda.transform(test_data)
#train_data = train_data[:,-11:]
#test_data = test_data[:,-11:]


from sklearn import linear_model, neighbors, svm, tree, preprocessing
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier, ExtraTreesClassifier

print ("Transform OK!")
print ("Let's train the model")
print("train_data",train_data.shape)
print("train_label",train_label.shape)


# AdaBoost
clf = AdaBoostClassifier(base_estimator=ExtraTreesClassifier(n_estimators=1000, verbose=1, min_samples_split=100,min_samples_leaf = 100), n_estimators=50,  learning_rate=0.8)
clf.fit(train_data, train_label)
result = clf.predict_proba(test_data)
#trian_data = preprocessing.normalize(train_data, norm='l2')
#test_data = preprocessing.normalize(test_data, norm='l2')

#clf1 = linear_model.LogisticRegression(random_state=1)
#clf2 = GradientBoostingClassifier(n_estimators=1500, #learning_rate=1.0,
#       verbose = 1, random_state=1).fit(train_data, train_label)
#clf3 = GaussianNB()
#eclf1 = VotingClassifier(estimators=[('lr', clf1), ('gb', clf2), ('gnb', clf3)], voting='soft')
#eclf1 = eclf1.fit(train_data, train_label)

#result = eclf1.predict_proba(test_data)

#eclf2 = VotingClassifier(estimators=[('lr', clf1), ('gb', clf2), ('gnb', clf3)], voting='soft')

#clf = ExtraTreesClassifier(n_estimators=10000, verbose=1)
#clf.fit(train_data, train_label)
#result = clf.predict_proba(test_data)
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
#result = clf.predict_proba(test_data)
#result = clf.predict_proba(test_data)

# Sort the result depending on deccending order of probability
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

