import xgboost as xgb
import numpy as np
from six.moves import cPickle as pickle
#from sklearn.model_selection import GridSearchCV
from sklearn import cross_validation, metrics
from xgboost.sklearn import XGBClassifier
import pandas as pd

# read in data
dataset = '/media/dat1/liao/dataset/liao_statis/'
train_data_filename = dataset + 'train_data_statis.pickle'
train_label_filename = dataset + 'train_label.pickle'
test_data_filename = dataset + 'test_data_statis.pickle'
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

def statis(data):
  month_mean = np.mean(data, axis=1)
  month_var = np.var(data, axis=1)
  month_min = np.amin(data, axis=1)
  month_max = np.amax(data, axis=1)
  month_range = np.ptp(data, axis=1)
  data = np.column_stack((data, month_mean, month_var, month_min, month_max, month_range))
  return data

#train_data = statis(train_data)
#test_data = statis(test_data)


print ('train: ', train_data.shape)
print ('test: ', test_data.shape)
#pca = PCA(50).fit(train_data)
#train_data = pca.transform(train_data)
#test_data = pca.transform(test_data)
#train_data = train_data[:, 365:]
#test_data = test_data[:, 365:]

#valid_data = train_data[1300:2000, :]
#valid_label = train_label[1300:2000, :]
#train_data = np.row_stack((train_data[:1300, :], train_data[2000:,:]))
#train_label = np.row_stack((train_label[:1300, :], train_label[2000:,:]))

xg_train = xgb.DMatrix( train_data, label=train_label, missing=-1 )
#xg_valid = xgb.DMatrix( valid_data, label=valid_label )
xg_test = xgb.DMatrix( test_data, missing=-1 )

# specify parameters via map
param = {'eta':0.1, 'gamma':0.1, 'subsample':0.8, 'colsample_bytree':0.8, 'scale_pos_weight': 6.04, 'objective':'binary:logistic'}

#param['eval_metric'] = ['rmse', 'logloss']

watchlist = [ (xg_train,'train') ] 

# train model
num_round = 0
print ('loading data end, start to boost trees')
#model = xgb.Booster({'nthread':4}) #init model
#model.load_model("eta0.1_subsample0.5_ams0.15.model") # load data
bst = xgb.train(param, xg_train, num_round, watchlist)#, xgb_model=model )
#xgb.plot_tree(bst, num_trees=2)
num_round = 1000

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
re = xgb.cv(param, xg_train, num_round, nfold=5, verbose_eval=100, metrics={'map', 'logloss'}, seed = 0, fpreproc = fpreproc)

print (re)

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
