import numpy as np
import pandas as pd

# pd.Series（data,index,dtype,name)
s1 = pd.Series([2.8, 3.01, 8.99, 8.59, 5.18])
s2 = pd.Series([2.8, 3.01, 8.99, 8.59, 5.18], index=['a', 'b', 'c', 'd', 'e'], name='这是一个series')
s3 = pd.Series({'北京': 2.8, '上海': 3.01, '广东': 8.99, '江苏': 8.59, '浙江': 5.18})
print(s1, s2, s3)

# pd.DataFrame(data,index,dtype,columns)
a = np.arange(1, 7).reshape(3, 2)
df1 = pd.DataFrame(a)
df2 = pd.DataFrame(a, index=['a', 'b', 'c'], columns=['x1', 'x2'])
df3 = pd.DataFrame({'x1': a[:, 0], 'x2': a[:, 1]})
x1 = {
    "张三": {'mysql': 88, 'python': 77, 'hive': 66},
    "李四": {'mysql': 11, 'python': 22, 'hive': 33}
}
df4 = pd.DataFrame(x1)
x2 = {
    "name": ['张三', '李四', '王五'],
    "age": [18, 22, 20],
    "sex": ["男", "女", "男"]
}
df5 = pd.DataFrame(x2)
print(df1, df2, df3, df4, df5)

# series和dataframe常用方法
df = pd.DataFrame(np.arange(1, 13).reshape(4, 3), index=['a', 'b', 'c', 'd'], columns=['x1', 'x2', 'x3'])
print(df.values)  # 返回对象所有元素的值

print(df.index)  # 返回行索引

print(df.dtypes)  # 返回索引

print(df.shape)  # 返回对象数据形状

print(df.ndim)  # 返回对象的维度

print(df.size)  # 返回对象的个数

print(df.columns)  # 返回列标签(只针对dataframe数据结构)

# os.chdir(r"C:\Users\xiaohong\Desktop\MCM小组\数据")   # 目的为更改文件路径
# pd.set_option('display.max_colunms',20)   # 设置最大显示列数
# pd.set_option('display.max_rows',100)   # 设置最大显示行数
# numbers = pd.read_csv('C:\\Users\\xiaohong\\Desktop\\数据项目组\\numbers_OCR_project.csv',nrows=100)   # 读取前100行
# numbers.to_csv('k22.csv',encoding='utf-8',index=False)   #index=False 表示不写出索引行,to_csv快速保存文件
#
# df1=pd.read_excel('meal_order_detail.xlsx',sheet_name = 'meal_order_detail')
# df1=pd.read_excel('meal_order_detail.xlsx',sheet_name = 0)
# df1.to_excel('al.xlsx',sheet_name = 'one',index=False) #保留数据


## 三、数据预处理

import pandas as pd

df_ex = pd.DataFrame({
    'brands': ['Yum Yum', 'Yum Yum', 'Indomie', 'Indomie', 'Indomie'],
    'style': ['cup', 'cup', 'cup', 'pack', 'pack'],
    'rating': [4, 4, 3.5, 15, 5]},
    index=['user1', 'user2', 'user3', 'user4', 'user5'],
    columns=['brands', 'style', 'rating'])
print(df_ex)

# 重复值处理
# a=pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\数据\\k.xlsx")
a = df_ex
print("是否存在重复观测：", any(a.duplicated()))  # 输出：True
a.drop_duplicates(inplace=True)  # 即当inplace=True时，选择返回更改后的数据，直接删除a中重复数据
f = pd.ExcelWriter('data1.xlsx')  # 创建新的文件对象
a.to_excel(f)
f.save()  # 保存文件.数据真正写入Excel文件当中

# 缺失值处理
# 缺失值检验
from numpy import NaN

data = pd.Series([10.0, None, 20, NaN, 30])
print(data.isnull())
print('是否存在缺失值：', any(data.isnull()))  # 输出：True

# 缺失值补全
# dropna(axis=0,how='any',thresh=None)
# #axis=0删除行，axis=1删除列
# #how的参数可选any或all，all表示删除全由NaN的行
# #thresh为整数类型，表示删除的条件，thresh=3表示一行中至少有3个非NaN值时才保留

# a=pd.read_excel("pdata2_33.xlsx",usecols=range(1,4))

a = df_ex
b1 = a.dropna()  # 删除所有的缺失值
b2 = a.dropna(axis=1, thresh=9)  # 删除有效数据个数小于9的列
b3 = a.drop('user4', axis=0)  # 删除user4的数据
print(b1, b2, b3)

df = pd.DataFrame({"id": [1001, 1002, 1003, 1004, 1005, 1006],
                   "date": pd.date_range('20130102', periods=6),
                   "city": ['Beijing ', 'SH', ' guangzhou ', 'Shenzhen', 'shanghai', 'BEIJING '],
                   "age": [23, 44, 54, 32, 34, 32],
                   "category": ['100-A', '100-B', '110-A', '110-C', '210-A', '130-F'],
                   "price": [1200, np.nan, 2133, 5433, np.nan, 4432]},
                  columns=['id', 'date', 'city', 'category', 'age', 'price'])
print(df)
df['city'].isnull()  # 查看一列空值
df['city'].unique()  # 查看某一列的唯一值
df.head()  # 默认前5行数据
df.tail()  # 默认后5行数据

# 1、用数字0填充空值：
# df.fillna(np.inf)
df['price'] = df['price'].fillna(0)

# 2、使用列prince的均值对NA进行填充：
df['price'] = df['price'].fillna(df['price'].mean())
# b=a.fillna(value={'gender':a.gender.mode()[0],   #性别使用众数替换
# 'age':a.age.interpolate(method='polynamial',order=2),   #年龄使用二次多项式插值替换
# 'income':a.income.interpolate()})   #收入使用线性插值替换

# #3、清除city字段的字符空格：
df['city'] = df['city'].map(str.strip)

# 4、大小写转换：
df['city'] = df['city'].str.lower()
print(df)
# 5、更改数据格式：
df['price'].astype('int')

# 6、更改列名称：
df.rename(columns={'category': 'category-size'})

# 7、删除后出现的重复值：
df['city'].drop_duplicates()

# 8 、删除先出现的重复值：
df['city'].drop_duplicates(keep='last')

# 9、数据替换：
df['city'].replace('sh', 'shanghai')

## 四、数据预处理
# 数据集成
# 两个DataFrame按列拼接

# pd.merge(
# 	left,right,how='inner',on=None,left_on=None,right_on=None,
# 	left_index=False,right_index=False,sort=False,suffixes=('_x','_y'),
# 	copy=True,indicator=False,validate=None)
# #两个DataFrame按行拼接
# pd.concat(
# 	objs.axis=0,join='outer',y,join_axes=None,
# 	ignore_index=False,keys=None,levels=None,names=None,
# 	verify_integrity=False,sort=None,copy=True)

df1 = pd.DataFrame({"id": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
                    "gender": ['male', 'female', 'male', 'female', 'male', 'female', 'male', 'female'],
                    "pay": ['Y', 'N', 'Y', 'Y', 'N', 'Y', 'N', 'Y', ],
                    "m-point": [10, 12, 20, 40, 40, 40, 30, 20]})
df_inner = pd.merge(df, df1, how='inner')  # 匹配合并，交集
df_left = pd.merge(df, df1, how='left')  #
df_right = pd.merge(df, df1, how='right')
df_outer = pd.merge(df, df1, how='outer')  # 并集
result = df1.append(df2)

# left = df
# right = df1
# result = left.join(right, on='key')
# pd.concat(df, axis=0, join='outer', join_axes=None, ignore_index=False,
# 	          keys=None, levels=None, names=None, verify_integrity=False,
# 	          copy=True)

# 去除异常值
# mu=a.counts.mean()    #计算平均值
# s=a.counts.std()   #计算标准差
# print("标准差法异常值上限检测："，any(a.counts>mu+2*s))  #输出：True
# print("标准差法异常值下限检测："，any(a.counts<mu-2*s>))   #输出：False
# Q1=a.counts.quantile(0.25) #计算下四分位数
# Q3=a.counts.quantile(0.75) #计算上四分位数
# IQR=Q3-Q1
# print("箱线图法异常值上限检测：",any(a.counts>Q3+1.5*IQR))   #输出：True
# print("箱线图法异常值下限检测：",any(a.counts<Q1-1.5*IQR))   #输出：False
# UB=Q3+1.5*IQR
# st=a.counts[a.counts<UB].max()
# print("判别异常值的上限临界值为：",UB)
# print("用于替换异常值的数据为：",st)
# a.loc[a.counts>UB,'counts']=st   #替换超过判别上限异常值

# 数据离散化处理
#
# 数据离散化处理即是将数据进行分箱，常用的分箱方法有等频分箱或者是等宽分箱。通常使用pd.cut或者pd.qcut函数进行操作。

pd.cut(x, bins, right=True, labels=None, retbins=False, precision=3, include_lowest=False)
# %注释：
#
# 1.x,类array对象，必须为一维，待切割的原形式
#
# 2.bins,整数、序列尺度、或间隔索引。若其为整数，则定义了x宽度分为内的等宽面元数量；若其为序列，则定义了允许非均因bin宽度的bin边缘，没有x范围的扩展
#
# 3.right,为布尔值。是否是左开右闭区间，right=True，表示左开右闭，right=False，表示左闭右开
#
# 4.labels，结果箱的标签，须于结果箱相同长度，如果False，只返回整数指标面元
#
# retbins,布尔值，是否返回面元
# 6.precision, 整数。返回面元的小数点几位
#
# 7.include_lowest,布尔值。第一个区间的左端点是否包含

pd.qcut(x, q, labels=None, retbins=False, precision=3, duplicates='raise')
# %注释：
#
# q为整数或分位数组成的数组
#
# 示例如下：

df['price_bin'] = pd.cut(df['price_new'], 5, labels=range(5))
df['Price_bin'].hist()
df['Price_new'].describe()  # 显示数据的基本信息
w = [100, 1000, 5000, 10000, 20000, 50000]
df['Price_bin'] = pd.cut(df['Price_new'], bins=w, labels=['低', '便宜', '划算', '中等', '高'], right=False)
df['Price_bin'].value_counts()
k = 5
w = [1.0 * i / k for i in range(k + 1)]
print(w)
df['Price_bin'] = pd.qcut(df['Price_new'], w, labels=range(k))
df['Price_bin'].hist()
k = 5
w1 = df['Price_new'].quantile([1.0 * i / k for i in range(k + 1)])  # 先计算分位数，再进行分段
w1[0] = w1[0] * 0.95
w[-1] = w1[1] * 1.05
print(w1)
df['Price_bin'] = pd.cut(df['Price_new'], w1, labels=range(k))
df['Price_bin'].hist()

# 特征选择
# # 1.哑编码
# # 对某一列数据进行pandas自带的(定性数据哑编码，定量数据二值化)
# pd.get_dummies(all['MSSubClass'],prefix='MSSubClass')
# # 2.卡方特征选择
# # 从已有的特征中选择出影响目标值最大的特征属性
# # { 分类：F统计量、卡方系数，互信息mutual_info_classif
# # { 连续：皮尔逊相关系数 F统计量 互信息 mutual_info_classif
# ch2 = pd.SelectKBest(chi2,k=10)
# X_train = pd.ch2.fit_transform(X_train, Y_train)X_test = pd.ch2.transform(X_test)
# print(ch2.get_support(indices=True))

# 3.PCA降维进行特征选择

pca = PCA(n_components=0.9)
X_train = pca.fit_transform(X_train)
X_test = transform(X_test)
# # 4.特征多项式扩展
# pf=PolynomialFeatures(degree=2, interaction_only=True, include_bias=True)
# x_train=pf.fit_transform(X_train)
# x_test=pf.transform(X_test)

##五、数据提取
# 主要用到的三个函数：loc,iloc和ix，loc函数按标签值进行提取，iloc按位置进行提取，ix可以同时按标签和位置进行提取。
data.iloc[:, [0]]  # 取第0列所有行，多取几列格式为 data.iloc[:,[0,1]]，取第0列和第1列的所有行
data.iloc[0]  # 第零行
# data[ 列名 ]： 取单列或多列，不能用连续方式取，也不能用于取行。
#
# data.列名： 只用于取单列，不能用于行。
#
# data[ i:j ]： 用起始行下标(i)和终止行下标(j)取单行或者连续多行，不能用于列的选取。
#
# data.loc[行名,列名]： 用对象的.loc[]方法实现各种取数据方式。
#
# data.iloc[行下标,列下标]： 用对象的.iloc[]方法实现各种取数据方式。


# 八、数据统计
# 数据采样，计算标准差，协方差和相关系数
#
# 1、简单的数据采样
df_inner.sample(n=3)

# 2、手动设置采样权重
weights = [0, 0, 0, 0, 0.5, 0.5]
df_inner.sample(n=2, weights=weights)

# 3、采样后不放回
df_inner.sample(n=6, replace=False)

# 4、采样后放回
df_inner.sample(n=6, replace=True)

# 5、 数据表描述性统计
df_inner.describe().round(2).T  # round函数设置显示小数位，T表示转置

# 6、计算列的标准差
df_inner['price'].std()

# 7、计算两个字段间的协方差
df_inner['price'].cov(df_inner['m-point'])

# 8、数据表中所有字段间的协方差
df_inner.cov()

# 9、两个字段的相关性分析
df_inner['price'].corr(df_inner['m-point'])  # 相关系数在-1到1之间，接近1为正相关，接近-1为负相关，0为不相关

# 10、数据表的相关性分析
df_inner.corr()

# 九、数据输出
# 分析后的数据可以输出为xlsx格式和csv格式

# 1、写入Excel
df_inner.to_excel('excel_to_python.xlsx', sheet_name='bluewhale_cc')

# 2、写入到CSV
df_inner.to_csv('excel_to_python.csv')
