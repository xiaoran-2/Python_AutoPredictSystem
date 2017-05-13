
# coding: utf-8

# In[1]:

import os
from os import listdir
from os.path import isfile, join
from os import walk
from bs4 import BeautifulSoup
from precessedData import *
from time import time



# In[2]:

#得到所有路径
mypath = ['1','2','3']
fs = []
for i in range(3):
    for (dirpath, dirnames, filenames) in walk(mypath[i]):
    #     print(dirpath)
    #     print(dirnames)
        for file in filenames:
            if file == 'export.css':
                continue
            fs.append(dirpath + "/" + file)
    #     break

#将文件目录写入
with open('filepath.txt', 'w') as f:
    for i in range(1,len(fs)):
        f.write(fs[i])

    f.close()
        


# In[3]:

index = [] #纪录所有fields.html的下标，按文件夹计算得分
for i in range(len(fs)):
    if fs[i].find('fields.html') >= 0:
        index.append(i)


# In[4]:

len(index)


# In[7]:

begin = time()
user_source_dict = {} # user_id : [总分，每一题的得分]
# 得到所有文件的评分
k = 0
# 得到部分文件进行测试，这里可以进行测试，但是我们的结果文件是全部的
# 测试的时候，可是设置n的值,小于长度即可

# n = len(index)
n = 10 
for i in range(0,n-1):
    if i% 50 == 0: print(i) #每50个，打印计算的进度
    user_id,source = uniteAll(fs[index[i]:index[i+1]])
    
    # tfidf得到关键词
    keyWords = getQuestonKeys(fs[index[i]])
    
    #得到用户答案的LDA矩阵
    QuestonLDAMat = getQuestonLDAMat(fs[index[i]])
    
    if user_id != None:
        #这里可以设置注释，得到需要的信息
#         user_source_dict[user_id] = [sum(source),source,keyWords]
        user_source_dict[user_id] = [sum(source),source,QuestonLDAMat]
        
    k = i
    
end = time()

print("Total time(minutes):", (end - begin)/60)


# In[8]:

# 写入文件分数和LDA矩阵
import csv
headers = ['user_id','总分','Question1分数','Question2分数','Question3分数','Question4分数','Question1 LDAMat','Question2 LDAMat','Question3 LDAMat','Question4 LDAMat']
with open('source_LDAMat.csv','w',newline="") as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    for key in user_source_dict.keys():
        tmp = []
        tmp.append(key)
        tmp.append(user_source_dict[key][0])
        tmp.extend(user_source_dict[key][1])
        for d in user_source_dict[key][2]:
            tmp.append(d)
        
        f_csv.writerows([tmp])



# In[7]:

# 写入文件分数和关键词
import csv
headers = ['user_id','总分','Question1分数','Question2分数','Question3分数','Question4分数','Question1KeyWords','Question2KeyWords','Question3KeyWords','Question4KeyWords']
with open('source_keywords.csv','w',newline="") as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    for key in user_source_dict.keys():
        tmp = []
        tmp.append(key)
        tmp.append(user_source_dict[key][0])
        tmp.extend(user_source_dict[key][1])
        for d in user_source_dict[key][2]:
            tmp.append(d)
        
        f_csv.writerows([tmp])



# In[7]:

# 写入文件,只有分数
import csv
headers = ['user_id','总分','Question1分数','Question2分数','Question3分数','Question4分数']
with open('user_source_dict.csv','w',newline="") as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    for key in user_source_dict.keys():
        tmp = []
        tmp.append(key)
        tmp.append(user_source_dict[key][0])
        tmp.extend(user_source_dict[key][1])
        f_csv.writerows([tmp])



# 处理结果文件，剔除含有空值的行，还有空值的原因是：有些数据不完整，例如没有评价文件，或者只有一个评分文件等
# 对于自动评分系统，Nan的评分，就相当与没有这个user的评分，可以直接删除，
# 
# 这里几个结果文件，是分批次运行处理的，因为全部运算太耗时，大概需要4个小时左右。
# 我们进行的整合，并给出的测试的代码
# 建议选择部分文件进行测试，

# In[130]:

# 处理结果文件，剔除含有空值的行，还有空值的原因是：有些数据不完整，例如没有评价文件，或者只有一个评分文件等
# 对于自动评分系统，Nan的评分，就相当与没有这个user的评分，可以直接删除，

import pandas as pd

source = pd.read_csv('user_source_dict.csv',error_bad_lines=False)
source1 = pd.read_csv('user_source_dict1.csv',error_bad_lines=False)
source2 = pd.read_csv('user_source_dict2.csv',error_bad_lines=False)
source3 = pd.read_csv('user_source_dict3.csv',error_bad_lines=False)


# In[125]:

source.info()
source1.info()
source2.info()
source3.info()


# In[131]:

source = source.append(source1)
source = source.append(source2)
source = source.append(source3)
# source.to_csv('user_source_dict.csv')
#删除空值，写入最后的文件
source = source.dropna()
source.info()


# In[132]:

source.to_csv('FinallySource.csv')


# In[123]:




# In[88]:



