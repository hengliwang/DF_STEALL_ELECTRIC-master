dataset = '/media/dat1/liao/dataset/'

user_data_filename = dataset + 'all_user_yongdian_data_2015.csv'
stretch_all_data_raw_filename = dataset + 'all_stretch_raw.csv'
stretch_all_data_filename = dataset + 'all_stretch.csv'

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
        line = [-1] * (3*DAYS_OF_YEAR)
        all_data[row[0]] = line
      f1 = to_float(row[2])
      f2 = to_float(row[3])
      f3 = to_float(row[4])
      if f1 > line[3*escape_days]:
        line[3*escape_days] = f1
      if f2 > line[3*escape_days+1]:
        line[3*escape_days+1] = f2
      if f3 > line[3*escape_days+2]:
        line[3*escape_days+2] = f3
      line_no += 1
      if not line_no % 1000000: print('solved ' + str(line_no) + ' lines now.')
  return all_data


def write_to_file(all_data, filename):
  with open(filename, 'wb') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE)
    for key in all_data:
      line = list()
      line.append(key)
      line.extend(all_data[key])
      writer.writerow(line) 

if __name__ == '__main__':
  print("Extract data from all user dataset now...")
  all_data = extract_data(user_data_filename)
  print("\nWrite raw data to file now...")
  write_to_file(all_data, dataset + 'all_three_columns_stretch.csv')
