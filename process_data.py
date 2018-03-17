import csv
import os
from six.moves import cPickle as pickle
import numpy as np
# import the package
#pickle: write the data by pickle

#This func transform the date 
#input: str
#output:str
#instance:2015/1/9 ---->20150109
def trans_date(date):
  new_date = date.split('/')##new_date is a list
  month = new_date[1]
  if len(month) == 1:
    new_month = '0'+ month
  else:
    new_month = month

  day = new_date[2]
  if len(day) == 1:
    new_day = '0' + day
  else:
    new_day = day

  new_date[1] = new_month
  new_date[2] = new_day
  return ('').join(new_date)#trans list to str

##next:
##combine the data with the same id
csvfile = file('/home/shaomingguang/shao_data/electric/all_data.csv')#change your path
read = csv.reader(csvfile)

all_data_list = list()
id_list = list()#just for search

for line in read:
  if line[0] == 'CONS_NO':#ignore the first line
    continue
  line[1] = trans_date(line[1])#trans the date
  if line[0] not in id_list:
    all_data_list.append(line)  
    id_list.append(line[0])
  else:
    ###cosume line not in all_data_list
    index = id_list.index(line[0])
    all_data_list[index].append(line[1])
    all_data_list[index].append(line[2])
    all_data_list[index].append(line[3])
    all_data_list[index].append(line[4])

### umtil now we have combine the data which has the same id
##all_data_list is a two demsion list
pickle_file = '/home/shaomingguang/shao_data/all_data.pickle'
f = open(pickle_file, 'wb')
save = all_data_list
pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
#####

###Let generate the train_data and train_label
pickle_file = '/home/shaomingguang/shao_data/electric/all_data.pickle'
f = open(pickle_file)
all_data_list = pickle.load(f)
f.close()
##remember all_data_list is a two demesion list
csvfile = file('/home/shaomingguang/shao_data/electric/train.csv')
read = csv.reader(csvfile)
train_data = list()
train_label = list()

not_find_num = 0
for line in read:
  flag = 0
  for temp in all_data_list:
    if temp[0]== line[0]:
      flag = 1
      train_label.append(line[1])
      train_data.append(temp)

  if flag == 0:
    not_find_num = not_find_num + 1
    print ("CAN NOT FIND")
    print (line[0])
    print (line[1])

print ("not find num",not_find_num)
pickle_file_data = '/home/shaomingguang/shao_data/electric/train_data.pickle'
pickle_file_label = '/home/shaomingguang/shao_data/electric/train_label.pickle'

f_data = open(pickle_file_data, 'wb')
f_label = open(pickle_file_label, 'wb')
save_data = train_data
save_label = train_label

pickle.dump(save_data, f_data, pickle.HIGHEST_PROTOCOL)
pickle.dump(save_label, f_label, pickle.HIGHEST_PROTOCOL)
f_data.close()
f_label.close()
#####

#####Let's generate the test_data

pickle_file = '/home/shaomingguang/shao_data/electric/all_data.pickle'
f = open(pickle_file)
all_data_list = pickle.load(f)
f.close()

csvfile = file('/home/shaomingguang/shao_data/electric/test.csv')
read = csv.reader(csvfile)
test_data = list()


not_find_num = 0
for line in read:
  flag = 0
  for temp in all_data_list:
    if temp[0]== line[0]:
      flag = 1
      test_data.append(temp)

  if flag == 0:
    not_find_num = not_find_num + 1
    print ("CAN NOT FIND")
    print (line[0])

print ("not find num",not_find_num)
pickle_file_data = '/home/shaomingguang/shao_data/electric/test_data.pickle'

f_data = open(pickle_file_data, 'wb')
save_data = test_data

pickle.dump(save_data, f_data, pickle.HIGHEST_PROTOCOL)
f_data.close()


###next: we sort the all_ data and delete the date information

###This func sort the data by date and delete the date information
### we do not felete the id infromation
def sort_del(temp):
  temp = temp[1:]#ignore usr_id
  temp_f = list()
  for i in temp:
    if i=='':
      temp_f.append(-1)
    else:
      temp_f.append(float(i))
  temp_f = np.array(temp_f)#change to np.array
  temp_f = temp_f.reshape(len(temp_f)/4,4)
  temp_arg = np.argsort(temp_f[:,0])
  temp_f = temp_f[temp_arg]
  temp_f = temp_f[:,1:]#delete the date information
  temp_f = temp_f.reshape(1,len(temp_f)*3)
  new_data = temp_f[0]###ok
 # new_data = list(new_data)
  return new_data

#####
##Let change the all_data
pickle_file_data = '/home/shaomingguang/shao_data/electric/all_data.pickle'
f_data = open(pickle_file_data)
all_data = pickle.load(f_data)
all_data_f = list()

for i in all_data:
  all_data_f.append(sort_del(i))

all_data_f = np.array(all_data_f)
pickle_file_data = '/home/shaomingguang/shao_data/electric/all_data_no_date.pickle'
f_data_save = open(pickle_file_data, 'wb')
save_data = all_data_f

pickle.dump(save_data, f_data_save, pickle.HIGHEST_PROTOCOL)
f_data.close()
f_data_save.close()

##Let's chage the train_data
pickle_file_data = '/home/shaomingguang/shao_data/electric/train_data.pickle'
f_data = open(pickle_file_data)
train_data = pickle.load(f_data)
train_data_f = list()

for i in train_data:
  train_data_f.append(sort_del(i))

train_data_f = np.array(train_data_f)
pickle_file_data = '/home/shaomingguang/shao_data/electric/train_no_date.pickle'
f_data_save = open(pickle_file_data, 'wb')
save_data = train_data_f
pickle.dump(save_data, f_data_save, pickle.HIGHEST_PROTOCOL)
f_data.close()
f_data_save.close()

##Let's change the test_data and generate the test_id
pickle_file_data = '/home/shaomingguang/shao_data/electric/test_data.pickle'
f_data = open(pickle_file_data)
test_data = pickle.load(f_data)
test_data_f = list()
test_data_id = list()

for i in test_data:
  test_data_f.append(sort_del(i))
  test_data_id.append(i[0])

test_data_f = np.array(test_data_f)
pickle_file_data = '/home/shaomingguang/shao_data/electric/test_no_date.pickle'
pickle_file_id = '/home/shaomingguang/shao_data/electric/test_id.pickle'
f_data_save = open(pickle_file_data, 'wb')
f_id_save = open(pickle_file_id,'wb')

save_id = test_data_id
save_data = test_data_f

pickle.dump(save_data, f_data_save, pickle.HIGHEST_PROTOCOL)
pickle.dump(save_id, f_id_save, pickle.HIGHEST_PROTOCOL)

f_data.close()
f_data_save.close()
f_id_save.close()

####
####important!!!!!
####The all_data_no_date train_no_date test_no_date is the data that without date information and sorted by date
####the type of three data type is ndarray
####but the data[i] is array([])







