使用pyhton3的进行开发，给予tfidf和LDA模型的两种计算方式。


filepath: 是所有文件的路径目录

output中间结果文件，因为运算时间过长，分开计算的：
user_source_dict.csv : 第1次的结果文件
user_source_dict1.csv : 第2次的结果文件
user_source_dict2.csv : 第3次的结果文件
user_source_dict3.csv : 第4次的结果文件
FinallySource.csv： 最有的结果，剔除了空值，给出每个用户的总分和各题的分数


jupyter notebook文件，可以用jupyter notebook运行（基于python3.5）
precessedData.ipynb ： 处理一个文件夹的文件
pingfen.ipynb ： 处理所有的文件夹

py文件：使用python3.5开发，可以直接运行的文件
precessedData.py ： 处理一个文件夹的文件
pingfen.py ： 处理所有的文件夹

注意：安装需要的包
bs4/BeautifulSoup
numpy
pandas
jiebe
gensim
注意看代码注释，运行的时候，使用小数据集测试即可,








