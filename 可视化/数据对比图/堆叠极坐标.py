##横向
import pyecharts.options as opts
from pyecharts.charts import Polar  # polar——极坐标系

x_attr = ["2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010"]
a = [25, 50.0, 50.5, 44.5, 53.5, 49, 54, 66, 59, 68.0, 54]
b = [24, 31.0, 26.0, 30.5, 38.0, 37, 52, 63, 59, 64.5, 43]
c = [22, 23.5, 25.5, 29.5, 32.0, 32, 37, 49, 42, 55.0, 37]

# 初始化
polar = (Polar()
         # add_schema: 加载模型中的地图 RadiusAxisOpts：极坐标半径配置项
         # AngleAxisOpt:极坐标角度配置项
         .add_schema(radiusaxis_opts=opts.RadiusAxisOpts(data=x_attr),
                     angleaxis_opts=opts.AngleAxisOpts(is_clockwise=True))

         # stack——实现数据堆叠，同个类目轴上系列配置相同的stack值可以堆叠放置
         # 配置不同的stack则会各自单独给出一个条目
         # type_ 根据极坐标图类型指定条形或扇形图形
         .add("第一名", a, type_="bar", stack="stack0")
         .add("第二名", b, type_="bar", stack="stack0")
         .add("第三名", c, type_="bar", stack="stack0")

         # set_global_opts:全局变量设置 TitleOpts:标题设置项
         .set_global_opts(title_opts=opts.TitleOpts(title="极坐标系-堆叠柱状图"))
         # render 会生成本地文件 HTML文件，默认会在当前目录生成 render.html文件
         # 也可传入路径参数，如 bar.render("mycharts.html")
         .render("极坐标-堆叠柱状图.html")

         )

##纵向
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import matplotlib as mpl
# 自定义坐标轴格式
# from matplotlib.ticker import FuncFormatter

# 导入数据
df = pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\k2.xlsx")  # 读取excel内数据
df.index = list("JFEDCBA")
# print(df) # 格式化输出

# 设置中文显示
plt.rcParams['font.sans-serif'] = 'SimHei'
# 解决负号无法正常显示的问题
plt.rcParams['axes.unicode_minus'] = False

N = 12  # 共有12个月
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)  # 获取12个月分类的角度值
width = np.pi / N  # 绘制扇型的宽度
labels = list(df.columns)  # 用第一行当坐标轴标签

# 开始绘图
fig = plt.figure(figsize=(8, 10))
fig = fig.gca(polar=True)
fig.set_theta_offset(np.pi / 2)

ax = plt.subplot(111, projection='polar')
label_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
for idx in df.index:
    # 每一行绘制一个扇形
    radii = df.loc[idx]  # 每一行数据
    ax.bar(theta, radii, width=0.4, bottom=0.0, label=idx, tick_label=labels)
    # plt.yticks([]) # 去掉y轴的坐标标签

    # ax.set_theta_zero_location(df.columns[0]) # 设置一月方向向上
    ax.set_theta_direction(-1)  # 顺时针方向绘图

plt.title('极坐标柱状堆叠图示例')
plt.legend(loc=4, bbox_to_anchor=(1.15, -0.07))  # 将label显示出来， 并调整位置
plt.show()
