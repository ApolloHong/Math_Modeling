import csv

import numpy as np
import pandas as pd
from mat4py import loadmat
from numpy import genfromtxt


##csv文件 读取
# csv to list
def loadCSV(filename):
    dataSet = []
    with open(filename, 'r') as file:
        csvReader = csv.reader(file)
        for line in csvReader:
            dataSet.append(line)
    return dataSet


# pandas header 每列的标签
test_df = pd.read_csv(r'filename', sep='\t', header=None)

# numpy 可以指定分隔符
my_data = genfromtxt('my_file.csv', delimiter=',')


##read csv
# csv
def csv_write(filename):
    with open(r'filename') as myFile:
        myWriter = csv.writer(myFile)
        # 输入list,写入一行
        myWriter.writerow([7, 'g'])
        myWriter.writerow([7, 'g'])
        # 写入两行
        myList = [[1, 2, 3], [4, 5, 6]]
        myWriter.writerows(myList)


# dat文件
def dat_file_ex():
    df = pd.read_csv(path='user.dat', sep='::', header=None,
                     names=['UserID', 'Gender', 'Age', 'Occupation', 'Zipcode'])
    c = np.fromfile('test.dat', dtype=int)


# txt文件
def txt_file_ex():
    a = np.loadtxt('test.txt')


# mat文件
def mat_file_ex():
    data = np.array(loadmat('test.mat'['dictKey'])).astype('float')

# ##特征数据处理
# data = test_df
# data.head()#显示前五行内容
# data.tail()#显示倒数五行内容
# print(data.shape,data.dtypes)#显示维度和查看每一列数据格式
# data.info()#查看数据表基本信息（维度，列表名，数据格式，所占空间)
# data.iloc[4]#输出第四行
# data.iloc[0:4]
# data.iloc[[0,2,4].[3,4]]#提取第0，2，4行的第3，4列
# #按月销量大小排序
# data = data.sort_values(by =['月销量'],ascending=False)
# #同时查看基本统计信息(最大值，最小值，平均值，中位数)
# data.describe()
#
# ##数据清洗
# #拆分'a事件'，变得标准化
# data['a事件'] = data['a事件'].str.split('',expend = True)[0]
# #
# data['收藏'] = data['收藏'].str.extract('(/d+)')
#
