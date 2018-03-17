#!/usr/bin/python

import numpy as np
from six.moves import cPickle as pickle

dataset = 'dataset/'
test_uids_filename = dataset + 'final_test_uids.pickle'
proba_1_filename = dataset + 'proba1.csv'
proba_2_filename = dataset + 'proba2.csv'

with open(test_uids_filename, 'rb') as f:
  test_uids = pickle.load(f)

proba1 = list()
with open(proba_1_filename, 'rb') as f:
  for p in f: proba1.append(float(p.rstrip()))
proba2 = list()
with open(proba_2_filename, 'rb') as f:
  for p in f: proba2.append(float(p.rstrip()))

result = np.array(proba1) + np.array(proba2) 

class Score:
  def __init__(self, uid, prob):
    self.uid = uid 
    self.prob = prob

scores = list()
for i in range(len(test_uid)):
  scores.append(Score(test_uid[i], result[i]))

scores = sorted(scores, key=lambda x: x.prob, reverse=True)

ff = open('merge_raw.csv', 'w')
for i in range(len(test_uid)):
    uid = scores[i].uid
    ff.write(uid + ',' + str(scores[i].prob) + '\n')
ff.close()

with open('merge_result.csv', 'w') as f:
  for i in range(len(test_uid)):
    f.write(scores[i].uid + '\n')


