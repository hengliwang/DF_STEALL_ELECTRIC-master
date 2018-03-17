#!/usr/bin/python
dataset = '/media/dat1/liao/fusai/dataset/'
destdir = '/media/dat1/liao/fusai/final_try/'

user_data_filename =  dataset + 'ALL_USER_YONGDIAN_DATA.csv'
train_set_filename = dataset + 'train.csv'
test_set_filename = dataset + 'test.csv'
dest_train_filename = destdir + 'train_data.csv'
dest_train_label_filename = destdir + 'train_label.csv'
dest_test_filename = destdir + 'test_data.csv'

import datetime
start_date = datetime.datetime(2014, 1, 1) # for calculating date delta

def transform_date(ori_date):
  """
     Transform the DATA_DATE from 'yyyy/mm/dd' to an integer stands
     for the day escape from 2015/1/1, for example, 2015/1/1 is 
     converted to 0, 2015/1/2 is converted to 1,... 
  
     Parameters:
     -----------
     ori_date: the date string which format is yyyy/mm/dd that need 
               to be converted

     Return:
     -------
     int: the escape days from start date
  """
  now_date =  datetime.datetime.strptime(ori_date, "%Y/%m/%d")
  return (now_date - start_date).days

def get_month(ori_date):
  now_date =  datetime.datetime.strptime(ori_date, "%Y/%m/%d")
  return (now_date.year - 2014) * 12 + now_date.month 


def to_float(value):
  x = -1
  try:
    x = float(value)
  except TypeError:
    pass
  except ValueError:
    pass
  except Exception, e:
    pass
  else:
    pass
  return x


import csv

DAYS_OF_YEAR = 1035
def extract_data(filename):
  all_data = dict()
  end_month_data = dict()
  start_month_data = dict()
  delta_month_data = dict()
  line_no = 0 # line number, for prompt
  with open(filename, 'rb') as csvfile:
    reader = csv.reader(csvfile)
    title = reader.next() # read title
    for row in reader:
      line = all_data.get(row[0], None)
      escape_days = transform_date(row[1])
      if line == None:
        line = [-1] * (DAYS_OF_YEAR * 3)
        all_data[row[0]] = line
      f_end = to_float(row[2])
      f_start = to_float(row[3])
      f_delta = to_float(row[4])
      if f_end > line[escape_days]: line[escape_days] = f_end
      if f_start > line[escape_days+DAYS_OF_YEAR]: 
        line[escape_days+DAYS_OF_YEAR] = f_start
      if f_delta > line[escape_days+2*DAYS_OF_YEAR]:
        line[escape_days+2*DAYS_OF_YEAR] = f_delta
      # solve month data
      month = get_month(row[1]) - 1

      month_line = end_month_data.get(row[0], None)
      if month_line == None:
        month_line = [None] * 34
        end_month_data[row[0]] = month_line
      if month_line[month] == None:
        month_line[month] = list()
      month_line[month].append(f_end)

      month_line = start_month_data.get(row[0], None)
      if month_line == None:
        month_line = [None] * 34
        start_month_data[row[0]] = month_line
      if month_line[month] == None:
        month_line[month] = list()
      month_line[month].append(f_start)
       
      month_line = delta_month_data.get(row[0], None)
      if month_line == None:
        month_line = [None] * 34
        delta_month_data[row[0]] = month_line
      if month_line[month] == None:
        month_line[month] = list()
      month_line[month].append(f_delta)

      line_no += 1
      if not line_no % 1000000: print('solved ' + str(line_no) + ' lines now.')
  return all_data, end_month_data, start_month_data, delta_month_data


def write_to_file(all_data, filename):
  with open(filename, 'wb') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE)
    for key in all_data:
      line = list()
      line.append(key)
      line.extend(all_data[key])
      writer.writerow(line) 


import numpy as np
from sklearn.preprocessing import Imputer


import math
def gene_train_set(train_filename, user_data, train_dest, month_data):
  train_uid = list()
  train_data = list()
  train_label = list()
  train_month_statis = list()
  with open(train_filename, 'rb') as rcsvfile:
    reader = csv.reader(rcsvfile)
    with open(train_dest, 'wb') as wcsvfile:
      writer = csv.writer(wcsvfile, quoting=csv.QUOTE_NONE)
      for row in reader:
        line = user_data.get(row[0], None)
        train_uid.append(row[0])
        train_data.append(line)
        train_label.append(int(row[1]))
        train_month_statis.append(month_data[row[0]])
        writer.writerow(line)
  with open(dest_train_label_filename, 'w') as f:
    for line in train_label:
      f.write(str(line) + '\n' )
  return train_uid, train_data, train_label, train_month_statis


def gene_test_set(test_filename, all_data, dest_filename, month_data):
  test_uid = list()
  test_data = list()
  test_month_statis = list()
  with open(test_filename, 'rb') as rcsvfile:
    reader = csv.reader(rcsvfile)
    with open(dest_filename, 'wb') as wcsvfile:
      writer = csv.writer(wcsvfile, quoting=csv.QUOTE_NONE)
      for row in reader:
        line = all_data.get(row[0], None)
        test_uid.append(row[0])
        test_data.append(line)
        test_month_statis.append(month_data[row[0]])
        writer.writerow(line)
  return test_uid, test_data, test_month_statis


from six.moves import cPickle as pickle
def write_to_pickle(data, filename):
  with open(filename, 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
  print("Extract data from all user dataset now...")
  all_data, end_month_data, start_month_data, delta_month_data = extract_data(user_data_filename)
  
  print ('convert month data...')
  month_data = dict()
  for user in end_month_data:
    end_month_line = end_month_data[user]
    start_month_line = start_month_data[user]
    delta_month_line = delta_month_data[user]
    month_data[user] = list()
    num_month = len(end_month_line)
    for i in range(num_month):
      if end_month_line[i] == None: month_data[user].extend([-1.]*5)
      else: month_data[user].extend([np.mean(end_month_line[i]),
                                    np.median(end_month_line[i]), 
                                    np.std(end_month_line[i]),
                                    np.var(end_month_line[i]), 
                                    np.max(end_month_line[i])])
      if start_month_line[i] == None: month_data[user].extend([-1.]*5)
      else: month_data[user].extend([np.mean(start_month_line[i]),
                                    np.median(start_month_line[i]), 
                                    np.std(start_month_line[i]),
                                    np.var(start_month_line[i]), 
                                    np.max(start_month_line[i])])
      if delta_month_line[i] == None: month_data[user].extend([-1.]*5)
      else: month_data[user].extend([np.mean(delta_month_line[i]),
                                    np.median(delta_month_line[i]), 
                                    np.std(delta_month_line[i]),
                                    np.var(delta_month_line[i]), 
                                    np.max(delta_month_line[i])])
      
  print("\nExtract train data now...")
  train_uid, train_data, train_label, train_month_statis = gene_train_set(train_set_filename, all_data, dest_train_filename, month_data)
  print("\nExtract test data now...")
  test_uid, test_data, test_month_statis = gene_test_set(test_set_filename, all_data, dest_test_filename, month_data)
  print("Done: file has been written to " + dest_test_filename)
  print("write pickle now...")
  write_to_pickle(np.array(train_month_statis), destdir + 'final_train_month_statis.pickle')
  write_to_pickle(np.array(test_month_statis), destdir + 'final_test_month_statis.pickle')
  write_to_pickle(np.array(train_data), destdir + 'final_train_data.pickle')
  write_to_pickle(np.array(train_label),destdir + 'final_train_labels.pickle')
  write_to_pickle(np.array(test_data), destdir+'final_test_data.pickle')
  write_to_pickle(np.array(test_uid), destdir+'final_test_uids.pickle')
  print("\nDone.")
