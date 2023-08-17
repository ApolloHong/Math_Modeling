import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Pie

# 准备数据
provinces = ['Americas', 'Europe', 'South-East Asia', 'Eastern Mediterranean', 'Western Pacific', 'Afica']
num = [1481995, 1172461, 596456, 443703, 553344, 147789]
color_series = ['#1E90FF', '#9ACD32', '#FFD700', '#FF4040', '#00CED1', '#87CEFA']

# 创建数据框
df = pd.DataFrame({'provinces': provinces, 'num': num})
# 降序排序
df.sort_values(by='num', ascending=False, inplace=True)

# 提取数据
v = df['provinces'].values.tolist()
d = df['num'].values.tolist()

# 实例化Pie类
pie1 = Pie(init_opts=opts.InitOpts(width='1350px', height='750px'))
# 设置颜色
pie1.set_colors(color_series)
# 添加数据，设置饼图的半径，是否展示成南丁格尔玫瑰图
pie1.add("", [list(z) for z in zip(v, d)],
         radius=["30%", "135%"],
         center=["50%", "65%"],
         rosetype="area"
         )
# 设置全局配置项
pie1.set_global_opts(title_opts=opts.TitleOpts(title='玫瑰图示例'),
                     legend_opts=opts.LegendOpts(is_show=False),
                     toolbox_opts=opts.ToolboxOpts())
# 设置系列配置项
pie1.set_series_opts(label_opts=opts.LabelOpts(is_show=True, position="inside", font_size=12,
                                               formatter="{b}:{c}天", font_style="italic",
                                               font_weight="bold", font_family="Microsoft YaHei"
                                               ),
                     )
# 生成html文档
pie1.render('南丁格尔玫瑰图.html')
