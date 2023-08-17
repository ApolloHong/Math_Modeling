import matplotlib.pyplot as plt
from pywaffle import Waffle

# 基础饼图
plt.figure(
    FigureClass=Waffle,
    rows=5,
    columns=10,
    values=[48, 40, 6, 6]
)
plt.show()

# # Prepare order data
# url_fea='C:\\Users\\hxt\\Desktop\\数模作业\\可视化作业模板\\数据统计图\\饼图\\华夫饼图\\features3.0.xlsx'
# url_ord='C:\\Users\\hxt\\Desktop\\数模作业\\可视化作业模板\\数据统计图\\饼图\\华夫饼图\\orders.xlsx'
# df_fea=pd.read_excel(url_fea)
# df_ord=pd.read_excel(url_ord)
#
# data={'workday':0,'holiday':0}
# col=6
# row=124
# for i in range(0,row+1,4):
#   if df_fea.loc[i,'节假日']==1:data['workday']+=sum(df_ord.iloc[i:i+4,col])
#   else: data['holiday']+=sum(df_ord.iloc[i:i+4,col])
# # print(data) #此行仅用于调试代码
#
# # Draw the waffle
# total = sum(data.values())
# fig=plt.figure(
#   FigureClass=Waffle,
#   rows = 10,
#   columns=10,
#   values = data,
#   # 设置title
#   title={
#     'label':'orders waffle of holiday vs workday',
#     'loc':'center',
#     'fontsize':12
#   },
#   # 设置图例
#    legend={
#     'labels': [f"{k} ({round(100*v/sum(list(data.values())),2)}%)" for k, v in data.items()],
#     'loc': 'lower left',
#     'bbox_to_anchor': (0, -0.2),
#     'ncol': 2,
#     'framealpha': 0,
#     'fontsize': 12
#   },
#   vertical=True
# )
# url_save='地址'
# fig.savefig(url_save)
# fig.show()
#
#
# #插入字典value
# plt.figure(
#     FigureClass=Waffle,
#     rows=5,
#     columns=10,
#     values={'Cat1': 20, 'Cat2': 12, 'Cat3': 8}, # 字典名字自动变为图例
#     # 设置图例的位置
#     legend={'loc': 'upper left', 'bbox_to_anchor': (1, 1)},
#     dpi=100
# )
# plt.show()


# 自带icon的图例
data = {'Evil': 48, 'Non-Evil': 46, 'Neutral': 3}
plt.figure(
    FigureClass=Waffle,
    rows=10,
    columns=10,
    values=data,
    colors=("#232066", "#983D3D", "#DCB732"),
    legend={'loc': 'upper left', 'bbox_to_anchor': (1, 1)},
    icons='rocket', icon_size=14,  # icon在网站上任意找
    dpi=300,
)
plt.show()

# 不同icon
plt.figure(
    FigureClass=Waffle,
    rows=5,
    values=[48, 46, 3],
    # 十六进制的颜色
    colors=["#FFA500", "#4384FF", "#C0C0C0"],
    # 指定晴天，阵雨，雪的图标
    icons=['sun', 'cloud-showers-heavy', 'snowflake'],
    font_size=12,
    icon_style='solid',
    icon_legend=True,
    legend={
        'labels': ['Sun', 'Shower', 'Snow'],
        'loc': 'upper left',
        'bbox_to_anchor': (1, 1)
    },
    dpi=120
)
plt.show()
