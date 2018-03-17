#!/usr/bin/python
dataset = '/media/dat1/liao/fusai/dataset/'
destdir = '/media/dat1/liao/fusai/final_try/dataset/'

user_data_filename =  dataset + 'ALL_USER_YONGDIAN_DATA.csv'
train_set_filename = dataset + 'train.csv'
test_set_filename = dataset + 'test.csv'

import datetime

def get_season(ori_date):
  now_date =  datetime.datetime.strptime(ori_date, "%Y/%m/%d")
  if now_date.month == 12: month = 0
  else: month = now_date.month
  return (now_date.year - 2014) * 4 + (month / 3) 


def to_float(value):
  x = -1
  try:
    x = float(value)
  except TypeError: pass
  except ValueError: pass
  except Exception, e: pass
  else: pass
  return x


import csv

def extract_data(filename):
  end_season_data = dict()
  start_season_data = dict()
  delta_season_data = dict()
  line_no = 0 # line number, for prompt
  with open(filename, 'rb') as csvfile:
    reader = csv.reader(csvfile)
    title = reader.next() # read title
    for row in reader:
      f_end = to_float(row[2])
      f_start = to_float(row[3])
      f_delta = to_float(row[4])
      # solve season data
      season = get_season(row[1])

      season_line = end_season_data.get(row[0], None)
      if season_line == None:
        season_line = [None] * 12
        end_season_data[row[0]] = season_line
      if season_line[season] == None:
        season_line[season] = list()
      season_line[season].append(f_end)

      season_line = start_season_data.get(row[0], None)
      if season_line == None:
        season_line = [None] * 12
        start_season_data[row[0]] = season_line
      if season_line[season] == None:
        season_line[season] = list()
      season_line[season].append(f_start)
       
      season_line = delta_season_data.get(row[0], None)
      if season_line == None:
        season_line = [None] * 12
        delta_season_data[row[0]] = season_line
      if season_line[season] == None:
        season_line[season] = list()
      season_line[season].append(f_delta)

      line_no += 1
      if not line_no % 1000000: print('solved ' + str(line_no) + ' lines now.')
  return end_season_data, start_season_data, delta_season_data


def write_to_file(all_data, filename):
  with open(filename, 'wb') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE)
    for key in all_data:
      line = list()
      line.append(key)
      line.extend(all_data[key])
      writer.writerow(line) 


import numpy as np
import math
def gene_train_set(train_filename, season_data):
  train_season_statis = list()
  with open(train_filename, 'rb') as rcsvfile:
    reader = csv.reader(rcsvfile)
    for row in reader:
      train_season_statis.append(season_data[row[0]])
  return train_season_statis


def gene_test_set(test_filename, season_data):
  test_season_statis = list()
  with open(test_filename, 'rb') as rcsvfile:
    reader = csv.reader(rcsvfile)
    for row in reader:
      test_season_statis.append(season_data[row[0]])
  return test_season_statis


from six.moves import cPickle as pickle
def write_to_pickle(data, filename):
  with open(filename, 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
  print("Extract data from all user dataset now...")
  end_season_data, start_season_data, delta_season_data = extract_data(user_data_filename)
  
  print ('convert season data...')
  season_data = dict()
  for user in end_season_data:
    end_season_line = end_season_data[user]
    start_season_line = start_season_data[user]
    delta_season_line = delta_season_data[user]
    season_data[user] = list()
    num_season = len(end_season_line)
   
    for i in range(num_season):
      if end_season_line[i] == None: season_data[user].extend([-1.]*5)
      else: season_data[user].extend([np.mean(end_season_line[i]),
                                    np.median(end_season_line[i]), 
                                    np.std(end_season_line[i]),
                                    np.var(end_season_line[i]), 
                                    np.max(end_season_line[i])])
      if start_season_line[i] == None: season_data[user].extend([-1.]*5)
      else: season_data[user].extend([np.mean(start_season_line[i]),
                                    np.median(start_season_line[i]), 
                                    np.std(start_season_line[i]),
                                    np.var(start_season_line[i]), 
                                    np.max(start_season_line[i])])
      if delta_season_line[i] == None: season_data[user].extend([-1.]*5)
      else: season_data[user].extend([np.mean(delta_season_line[i]),
                                    np.median(delta_season_line[i]), 
                                    np.std(delta_season_line[i]),
                                    np.var(delta_season_line[i]), 
                                    np.max(delta_season_line[i])])
      
  print("\nExtract train data now...")
  train_season_statis = gene_train_set(train_set_filename,  season_data)
  print("\nExtract test data now...")
  test_season_statis = gene_test_set(test_set_filename, season_data)
  print("write pickle now...")
  write_to_pickle(np.array(train_season_statis), destdir + 'final_train_season_statis.pickle')
  write_to_pickle(np.array(test_season_statis), destdir + 'final_test_season_statis.pickle')
  print("\nDone.")
