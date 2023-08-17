import matplotlib
import numpy as np
from matplotlib import pyplot as plt

font = matplotlib.font_manager.FontProperties(fname="C\\Source", size=18)

properties = ['输出', 'KDA', '发育', '团战', '生存']
values = [40, 91, 44, 90, 95, 40]
theta = np.linspace(0, np.pi * 2, 6)
plt.polar(theta, values)
plt.xticks(theta, properties, fontproperties=font)
plt.fill(theta, values)

##详情看网站
