dataset = '/media/dat1/liao/dataset/new_new_try/'

user_data_filename =  '/media/dat1/liao/dataset/all_user_yongdian_data_2015.csv'
stretch_all_data_raw_filename = dataset + 'all_stretch_raw.csv'
stretch_all_data_filename = dataset + 'all_stretch.csv'
stretch_interp_all_data_filename = dataset + 'all_interp_stretch.csv'
statistics_all_data_filename = dataset + 'all_statis_stretch.csv'
train_set_filename = '/media/dat1/liao/dataset/train.csv'
test_set_filename = '/media/dat1/liao/dataset/test.csv'
dest_train_filename = dataset + 'train__data.csv'
dest_test_filename = dataset + 'test__data.csv'

import datetime
start_date = datetime.datetime(2015, 1, 1) # for calculating date delta

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

DAYS_OF_YEAR = 365
def extract_data(filename):
  all_data = dict()
  line_no = 0 # line number, for prompt
  with open(filename, 'rb') as csvfile:
    reader = csv.reader(csvfile)
    title = reader.next() # read title
    for row in reader:
      line = all_data.get(row[0], None)
      escape_days = transform_date(row[1])
      if line == None:
        line = [0] * DAYS_OF_YEAR + [-1] * DAYS_OF_YEAR
        all_data[row[0]] = line
      f_start = to_float(row[3])
      f_delta = to_float(row[4])
      if f_start > line[escape_days]:
        line[escape_days] = f_start
      if f_delta > line[escape_days+DAYS_OF_YEAR]:
        line[escape_days+DAYS_OF_YEAR] = f_delta
      line_no += 1
      if not line_no % 1000000: print('solved ' + str(line_no) + ' lines now.')
  return all_data


THRESHOLD = 100
def convert_data(all_data):
  for key in all_data:
    line = all_data[key]
    base = 0
    max_v = line[0]
    for i in range(1,DAYS_OF_YEAR):
      if line[i] != 0.0 and line[i] != -1:
        if line[i] + base > max_v: max_v = line[i] + base
        elif line[i] + base < max_v - THRESHOLD:
          base = max_v
          max_v = line[i] + base
        line[i] += base


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
def do_interp(all_data):
  for key in all_data:
    value = all_data[key]
    x = list()
    xp = list()
    fp = list()
    for i in range(DAYS_OF_YEAR): # start_value
      if value[i] == -1 or value[i] == 0: x.append(i)
      else:
        xp.append(i)
        fp.append(value[i])
    if len(x) != 0 and len(xp) != 0:
      y = np.interp(x, xp, fp)
      for i in range(len(x)):
        value[x[i]] = y[i]
    # delta value: impute with most frequence value
    delta_value = value[DAYS_OF_YEAR:]
    if sum(np.array(delta_value) == -1) != len(delta_value):
      imp = Imputer(missing_values=-1, strategy='most_frequent', axis=1)
      delta_value = imp.fit(delta_value).transform(delta_value)
      value[DAYS_OF_YEAR:] = delta_value[0][:].tolist()


import math
def gene_train_set(train_filename, user_data, train_dest):
  train_uid = list()
  train_data = list()
  train_label = list()
  with open(train_filename, 'rb') as rcsvfile:
    reader = csv.reader(rcsvfile)
    with open(train_dest, 'wb') as wcsvfile:
      writer = csv.writer(wcsvfile, quoting=csv.QUOTE_NONE)
      for row in reader:
        line = user_data.get(row[0], None)
        if line == None or math.fsum(line) < 0.0:
          pass
        else:
          train_uid.append(row[0])
          train_data.append(line)
          train_label.append(int(row[1]))
          writer.writerow(line)
  return train_uid, train_data, train_label


def gene_test_set(test_filename, all_data, dest_filename):
  test_uid = list()
  test_data = list()
  do_not_use_uid = list()
  with open(test_filename, 'rb') as rcsvfile:
    reader = csv.reader(rcsvfile)
    with open(dest_filename, 'wb') as wcsvfile:
      writer = csv.writer(wcsvfile, quoting=csv.QUOTE_NONE)
      for row in reader:
        line = all_data.get(row[0], None)
        if line == None:# or math.fsum(line) <= 0.0:
          do_not_use_uid.append(row[0])
        else:
          test_uid.append(row[0])
          test_data.append(line)
          writer.writerow(line)
  with open(dataset+'test_no_use.csv', 'w') as f:
    for i in range(len(do_not_use_uid)):
      f.write(do_not_use_uid[i] + '\n')
  return test_uid, test_data


from six.moves import cPickle as pickle
def write_to_pickle(data, filename):
  with open(filename, 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def add_statis(data):
  all_data = np.array(data)
  # start value
  start_mean_value = np.mean(all_data[:DAYS_OF_YEAR], axis=1)
  start_min_value = np.amin(all_data[:DAYS_OF_YEAR], axis=1)
  start_max_value = np.amax(all_data[:DAYS_OF_YEAR], axis=1)
  start_range_value = np.ptp(all_data[:DAYS_OF_YEAR], axis=1)
  start_quart_value = np.percentile(all_data[:DAYS_OF_YEAR], 25, axis=1)
  start_half_value = np.percentile(all_data[:DAYS_OF_YEAR], 50, axis=1)
  start_fif_value = np.percentile(all_data[:DAYS_OF_YEAR], 75, axis=1)
  start_median_value = np.median(all_data[:DAYS_OF_YEAR], axis=1)
  start_std_value = np.std(all_data[:DAYS_OF_YEAR], axis=1)
  start_var_value = np.var(all_data[:DAYS_OF_YEAR], axis=1)
  start_special_value = start_mean_value / start_std_value
  # delta value
  delta_mean_value = np.mean(all_data[DAYS_OF_YEAR:], axis=1)
  delta_min_value = np.amin(all_data[DAYS_OF_YEAR:], axis=1)
  delta_max_value = np.amax(all_data[DAYS_OF_YEAR:], axis=1)
  delta_range_value = np.ptp(all_data[DAYS_OF_YEAR:], axis=1)
  delta_quart_value = np.percentile(all_data[DAYS_OF_YEAR:], 25, axis=1)
  delta_half_value = np.percentile(all_data[DAYS_OF_YEAR:], 50, axis=1)
  delta_fif_value = np.percentile(all_data[DAYS_OF_YEAR:], 75, axis=1)
  delta_median_value = np.median(all_data[DAYS_OF_YEAR:], axis=1)
  delta_std_value = np.std(all_data[DAYS_OF_YEAR:], axis=1)
  delta_var_value = np.var(all_data[DAYS_OF_YEAR:], axis=1)
  delta_special_value = delta_mean_value / delta_std_value
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
  all_data = np.column_stack((all_data, start_special_value))
  
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
  all_data = np.column_stack((all_data, delta_special_value))
  return all_data.tolist()


if __name__ == '__main__':
  print("Extract data from all user dataset now...")
  all_data = extract_data(user_data_filename)
  print("\nWrite raw data to file now...")
  write_to_file(all_data, stretch_all_data_raw_filename)
  print("\nConvert data of all user now...") 
  convert_data(all_data)
  print("\nWrite data to file now...") 
  write_to_file(all_data, stretch_all_data_filename)
  print("\nDo interplotion for missing value...")
  do_interp(all_data)
  print("\nWrite interplotion data to file now...")
  write_to_file(all_data, stretch_interp_all_data_filename)
  print("\nExtract train data now...")
  train_uid, train_data, train_label = gene_train_set(train_set_filename, all_data, dest_train_filename)
  print("\nExtract test data now...")
  test_uid, test_data = gene_test_set(test_set_filename, all_data, dest_test_filename)
  print("Done: file has been written to " + dest_test_filename)
  print("write pickle now...")
  write_to_pickle(np.array(train_data), dataset + 'train_data.pickle')
  write_to_pickle(np.array(train_label),dataset + 'train_label.pickle')
  write_to_pickle(np.array(test_data), dataset+'test_data.pickle')
  write_to_pickle(test_uid, dataset+'test_uid.pickle')
  print("\nDone.")
