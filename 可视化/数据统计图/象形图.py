import json

from pyecharts import options as opts
from pyecharts.charts import PictorialBar

with open("C:\\Users\\Apollo\\Pictures\\Screenshots\\屏幕截图_20230104_134747.png", "r", encoding="utf-8") as f:
    symbols = json.load(f)

c = (
    PictorialBar()
    .add_xaxis(["1", "2", "3", "4", "5"])  # x轴设置
    .add_yaxis(
        "114.514",
        [
            {"value": 1.14, "symbol": symbols["reindeer"]},  # 每一行的数据直接改动value后的数值即可，自定义的图形
            {"value": 5.14, "symbol": symbols["ship"]},  # 只需要替换“symbol”：后面的对象。
            {"value": 1, "symbol": symbols["plane"]},
            {"value": 1.45, "symbol": symbols["train"]},
            {"value": 1.4, "symbol": symbols["car"]},
        ],
        label_opts=opts.LabelOpts(is_show=False),  # 是否显示标签
        symbol_size=22,  # 图像大小
        symbol_repeat="fixed",
        symbol_offset=[0, 5],
        is_symbol_clip=True,
    )
    .add_yaxis(
        "1919.810",
        [
            {"value": 1.91, "symbol": symbols["reindeer"]},
            {"value": 9.8, "symbol": symbols["ship"]},
            {"value": 1.01, "symbol": symbols["plane"]},
            {"value": 9.19, "symbol": symbols["train"]},
            {"value": 8.10, "symbol": symbols["car"]},
        ],
        label_opts=opts.LabelOpts(is_show=False),
        symbol_size=22,
        symbol_repeat="fixed",
        symbol_offset=[0, -25],
        is_symbol_clip=True,
    )
    .reversal_axis()  # 反转横纵轴
    .set_global_opts(
        title_opts=opts.TitleOpts(title="PictorialBar-Vehicles in X City"),  # 图的标题
        xaxis_opts=opts.AxisOpts(is_show=False),
        yaxis_opts=opts.AxisOpts(
            axistick_opts=opts.AxisTickOpts(is_show=False),
            axisline_opts=opts.AxisLineOpts(
                linestyle_opts=opts.LineStyleOpts(opacity=0)
            ),
        ),
    )
    .render("1.html")  # 输出到指定文件，可以更改
)
