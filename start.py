#!/usr/bin/python
import xgboost as xgb
import csv
import numpy as np
from six.moves import cPickle as pickle

# read in data
dataset = 'dataset/'
train_data_filename = dataset + 'final_train_data.pickle'
train_label_filename = dataset + 'final_train_labels.pickle'
test_data_filename = dataset + 'final_test_data.pickle'
test_uid_filename = dataset + 'final_test_uids.pickle'
tr_train_uid_filename = dataset + 'final_train_idf.pickle'
tr_test_uid_filename = dataset + 'final_test_idf.pickle'

with open(train_data_filename, 'rb') as f:
  train_data = pickle.load(f)
with open(train_label_filename, 'rb') as f:
  train_label = pickle.load(f)
with open(test_data_filename, 'rb') as f:
  test_data = pickle.load(f)
with open(test_uid_filename, 'rb') as f:
  test_uid = pickle.load(f)
with open(tr_train_uid_filename, 'rb') as f:
  tr_train_uid = pickle.load(f)
with open (tr_test_uid_filename, 'rb') as f:
  tr_test_uid = pickle.load(f)

from sklearn.decomposition import PCA

print ('train: ', train_data.shape)
print ('test: ', test_data.shape)

print ('construct train & test data...')

for i in range(len(train_data)):
  for j in range(len(train_data[i])):
    if train_data[i][j] != train_data[i][j]: train_data[i][j] = -1

train_data = np.column_stack((train_data, tr_train_uid))
test_data = np.column_stack((test_data, tr_test_uid))

print ('train: ', train_data.shape)
print ('test: ', test_data.shape)


xg_train = xgb.DMatrix( train_data, label=train_label, missing=-1 )
xg_test = xgb.DMatrix( test_data, missing=-1 )

label = xg_train.get_label()
ratio = float(np.sum(label == 0)) / np.sum(label==1)

# specify parameters via map
param = {'eta':0.005, 'silent':1, 'min_child_weight':0, 'gamma':0, 'subsample':0.9, 'lambda':0.9, 'colsample_bytree':0.85, 'scale_pos_weight': ratio, 'objective':'binary:logistic'}
#param = {'eta':0.005, 'silent':1, 'min_child_weight':0, 'gamma':0, 'subsample':0.8, 'lambda':0.8, 'colsample_bytree':0.9, 'scale_pos_weight': ratio, 'objective':'binary:logistic'}

#param['eval_metric'] = ['rmse', 'logloss']

watchlist = [ (xg_train,'train') ] 

# train model
num_round = 8000 
print ('loading data end, start to boost trees')
#model = xgb.Booster({'nthread':4}) #init model
#model.load_model("eta0.1_subsample0.5_ams0.15.model") # load data

def lr(boosting_round, num_boost_round):
  if boosting_round >= 6000:
    return 0.0005
  if boosting_round >= 3000:
    return 0.0001
  return 0.005

bst = xgb.train(param, xg_train, num_round, watchlist,  verbose_eval=20 )#, xgb_model=model )

with open('feature_importance.csv', 'wb') as fout:
  scores = bst.get_fscore()
  #s = sorted(scores.iteritems(), key=lambda x:x[1], reverse=True)
  for line in scores:
    fout.write(line+','+str(scores[line])+'\n')
#bst.dump_model(fout, with_stats=True)

#xgb.plot_tree(bst, num_trees=2)
num_round = 0

print ('running cross validation, with preprocessing function')

# define the preprocessing function
# used to return the preprocessed training, test data, and parameter
# we can use this to do weight rescale, etc.
# as a example, we try to set scale_pos_weight
def fpreproc(dtrain, dtest, param):
  label = dtrain.get_label()
  ratio = float(np.sum(label == 0)) / np.sum(label==1)
  param['scale_pos_weight'] = ratio
  #wtrain = dtrain.get_weight()
  #wtest = dtest.get_weight()
  #sum_weight = sum(wtrain) + sum(wtest)
  #wtrain *= sum_weight / sum(wtrain)
  #wtest *= sum_weight / sum(wtest)
  #dtrain.set_weight(wtrain)
  #dtest.set_weight(wtest)
  return (dtrain, dtest, param)

# do cross validation, for each fold
# the dtrain, dtest, param will be passed into fpreproc
# then the return value of fpreproc will be used to generate
# results of that fold
re = xgb.cv(param, xg_train, num_round, nfold=5, verbose_eval=100, metrics={'map'}, seed = 0, fpreproc = fpreproc)

print(re)
re.to_csv('cv.csv', encoding='utf-8', index=True)

# save out model
#bst.save_model('eta0.1_subsample0.5_ams0.15.model')
# make prediction
result = bst.predict(xg_test)


print (result)

class Score:
  def __init__(self, uid, prob):
    self.uid = uid
    self.prob = prob

scores = list()
for i in range(len(test_uid)):
  scores.append(Score(test_uid[i], result[i]))

scores = sorted(scores, key=lambda x: x.prob, reverse=True)

ff = open('xgb_raw.csv', 'w')
for i in range(len(test_uid)):
    uid = scores[i].uid
    ff.write(uid + ',' + str(scores[i].prob) + '\n')
ff.close()

with open('xgb_result.csv', 'w') as f:
  for i in range(len(test_uid)):
    f.write(scores[i].uid + '\n')

print (result)
with open('all_data_proba.csv', 'wb') as f:
  for line in result: f.write(str(line) + '\n')
