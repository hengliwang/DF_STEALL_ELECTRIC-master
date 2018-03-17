"""
--------------------------------------------------------------
Extract data process

Detail: Extract all users' information such as, DATA_DATE, 
        KWH_READING, KWH_READING1, KWH in train dataset and
        do some transform from all users' information file. 

Author: LIAO

Version: V1.1 2016/10/5

Copyright(c) 2016, NCLAB All rights reserved
--------------------------------------------------------------

"""

dataset = '/media/dat1/liao/dataset/'

# In File: uers' electric data file
user_data_filename = dataset + 'all_user_yongdian_data_2015.csv'
train_data_filename = dataset + 'train.csv'
test_data_filename = dataset + 'test.csv'

# Out File: data of users from all user electirc file
extracted_train_user_data_filename = dataset + 'train_data.csv'
extracted_test_user_data_filename = dataset + 'test_data.csv'

stretch_train_data_filename = dataset + 'train_stretch.csv'
stretch_test_data_filename = dataset + 'test_stretch.csv'

import csv


def get_data_and_label(filename):
  """ Get data to label couple from csv file with given filename """
  data2label = dict()
  # read csv file
  with open(filename, 'rb') as csvfile:
    reader = csv.reader(csvfile)
    # train/val data which must contains data and label
    for row in reader:
       data2label[row[0]] = row[1] if len(row) > 1 else ''
  return data2label 


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


def extract_to_file(data2label, info_filename, dest_filename):
  """ 
     Extract all users' data in data2label from file 'filename',
     transform the date(@see #transform_date), and write data 
     to relative file.
    
     Parameters:
     -----------
     data2label: user id to label map
     info_filename: filename of all user's data file
     dest_filename: destination filename which extracted data 
                    should writed to 
  """
  with open(info_filename, 'rb') as rcsvfile:
    reader = csv.reader(rcsvfile)
    with open(dest_filename, 'wb') as wcsvfile:
      writer = csv.writer(wcsvfile, quoting=csv.QUOTE_NONE)
      title = reader.next() # include title
      ''' search for my users' data
          row[0-4] stands for: 0-CONS_NO 1-DATA_DATE 2-KWH_READING
          3-KWH_READING1 4-KWH, add some if you need, remember modify
          both title and content
      ''' 
      writer.writerow([title[0], title[1], title[2], title[4], 'LABEL']) # write title
      line_no = 0 # line number, for prompt
      for row in reader:
        if row[0] in data2label:
          line = [row[0], transform_date(row[1]), row[2], row[4], data2label[row[0]]]
          writer.writerow(line)
        line_no += 1
        if not line_no % 1000000: print('solved ' + str(line_no) + ' lines now.')


DAYS_OF_YEAR = 365
def extract_data(filename, data2label):
  """
     Extract data to an array, which stretch one user to a line
  """
  user_data = dict()
  line_no = 0 # line number, for prompt
  user_dup = dict()
  with open(filename, 'rb') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
      if row[0] in data2label:
        user_line = user_data.get(row[0], None)
        escape_days = transform_date(row[1])
        if user_line == None:
          user_line = [-1] * DAYS_OF_YEAR
          user_data[row[0]] = user_line
        if user_line[escape_days] != -1:
          if row[0] in user_dup: user_dup[row[0]] = user_dup[row[0]] + 1
          else: user_dup[row[0]] = 1
          #print ("Warning: "+row[0]+" duplicate data of date: "+row[1]+" label:"+data2label[row[0]])
        try:
          user_line[escape_days] = float(row[2]) # just today's end value
        except TypeError:
          pass
        except ValueError:
          pass
        except Exception, e:
          pass
        else:
          pass
      line_no += 1
      if not line_no % 1000000: print('solved ' + str(line_no) + ' lines now.')
#  for key in user_dup:
#    print ("User " + key + ": " + str(user_dup[key]))
#  print ("total number of dup: " + str(len(user_dup)))
  return user_data
        

def write_data(user_data, data2label, filename):
  with open(filename, 'wb') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE)
    for key in user_data:
      user_line = user_data[key]
      line = list()
      line.append(key)
      for i in user_line: line.append(str(i))
      line.append(data2label[key])
      writer.writerow(line)
    

if __name__ == "__main__":
  print ("extracted users' infos of train data now...")
  data2label=get_data_and_label(train_data_filename)
  ''' Uncomment to extract train/test data and write to file '''
  #extract_to_file(data2label, user_data_filename, extracted_train_user_data_filename) 
  #print ("Over: user data are stored in " + extracted_train_user_data_filename)
  user_data = extract_data(user_data_filename, data2label)
  write_data(user_data, data2label, stretch_train_data_filename)


  print ("extracted users' infos of test data now...")
  data2label=get_data_and_label(test_data_filename)
  user_data = extract_data(user_data_filename, data2label)
  write_data(user_data, data2label, stretch_test_data_filename)
  #data2label=get_data_and_label(test_data_filename)
  #extract_to_file(data2label, user_data_filename, extracted_test_user_data_filename) 
  #print ("Over: user data are stored in " + extracted_test_user_data_filename)
  
