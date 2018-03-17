from __future__ import print_function
import tensorflow as tf
import numpy as np
from six.moves import cPickle as pickle
from sklearn.decomposition import PCA
from sklearn import preprocessing
import random

dataset = '/media/dat1/liao/dataset/new_new_try/'
train_data_filename = dataset + 'train_data_statis.pickle'
train_label_filename = dataset + 'train_label.pickle'
test_data_filename = dataset + 'test_data_statis.pickle'
test_uid_filename = dataset + 'test_uid.pickle'

# load dataset 
print('load dataset...')
with open(train_data_filename, 'rb') as f:
  train_data = pickle.load(f)
with open(train_label_filename, 'rb') as f:
  train_label = pickle.load(f)
with open(test_data_filename, 'rb') as f:
  test_data = pickle.load(f)
with open(test_uid_filename, 'rb') as f:
  test_uid = pickle.load(f)
print ('Training set', train_data.shape, train_label.shape)
print ('Test set', test_data.shape)#, test_uid.shape)

n_classes = 2

shulf = np.column_stack((train_data, train_label))
random.shuffle(shulf)
train_data = shulf[:,:-1]
train_label = shulf[:,-1]


print('reformat dataset...')
pca = PCA(300).fit(train_data)
train_data = pca.transform(train_data)
test_data = pca.transform(test_data)


n_samples, n_features = train_data.shape

train_label = (np.arange(n_classes) == train_label[:,None]).astype(np.float32)
#test_label = (np.arange(n_classes) == test_label[:,None]).astype(np.float32)
test_data = test_data.astype(np.float32)

#train_subset = range(0,1200) + range(2800, 9900)
#valid_subset = range(1200, 2800)

#train_data = train_data[train_subset]
#train_label = train_label[train_subset]
#valid_data = train_data[valid_subset]
#valid_label = train_label[valid_subset]
#tf_test_data = tf.constant(test_data)

train_data = preprocessing.normalize(train_data, norm='l2')
#valid_data = preprocessing.normalize(valid_data, norm='l2')


print ('Training set', train_data.shape, train_label.shape)
#print ('Validation set', valid_data.shape, valid_label.shape)
print ('Test set', test_data.shape)#, test_uid.shape)

# constant 
learning_rate = 0.001
num_steps = 200000
test_inter = 200
hidden1_units = 100
hidden2_units = 32
batch_size = 100


def ip_layer(x, W, b):
  return tf.matmul(x, W) + b


graph = tf.Graph()
with graph.as_default():

  x = tf.placeholder(tf.float32, [None, n_features])
  y_ = tf.placeholder(tf.float32, [None, n_classes])   
 
  # hidden 1
  W1 = tf.Variable(
    tf.truncated_normal([n_features, hidden1_units]))
  b1 = tf.Variable(
    tf.constant(0.1, shape=[hidden1_units]))
  hidden1 = tf.nn.relu(ip_layer(x, W1, b1))
  test_h1 = tf.nn.relu(ip_layer(test_data, W1, b1))
  
  # hidden 2
#  W2 = tf.Variable(
#    tf.truncated_normal([hidden1_units, hidden2_units]))
#  b2 = tf.Variable(
#    tf.constant(0.1, shape=[hidden2_units]))
#  hidden2 = tf.nn.relu(ip_layer(hidden1, W2, b2))

  # dropout
  keep_prob = tf.placeholder("float")  
  h2_drop = tf.nn.dropout(hidden1, keep_prob)  
  
  # output layer
  W3 = tf.Variable(tf.truncated_normal([hidden1_units, n_classes]))
  b3 = tf.Variable(tf.constant(0.1, shape=[n_classes]))
  output = tf.nn.relu(ip_layer(h2_drop, W3, b3))
  test_output = tf.nn.relu(ip_layer(test_h1, W3, b3))  

  y = tf.nn.softmax(output)
  test_prob = tf.nn.softmax(test_output)  

  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(output, y_))
  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))  
  a_accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))   


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_label.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_data[offset:(offset + batch_size), :]
    batch_labels = train_label[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {x : batch_data, y_ : batch_labels, keep_prob : 0.5}
    _, l, predictions = session.run(
      [optimizer, loss, y], feed_dict=feed_dict)
    if (step % test_inter == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))  
 
  print(session.run(test_prob))
  result  = session.run(test_prob)
  class Score:
    def __init__(self, uid, prob):
      self.uid = uid
      self.prob = prob

  print(result)

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

