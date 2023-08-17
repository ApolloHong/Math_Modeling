import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv("C:\\Users\\xiaohong\\Desktop\\数据项目组\\numbers_OCR_project.csv")

df = np.eye(10)
# df.head()
sns.heatmap(df, annot=True)
plt.show()
