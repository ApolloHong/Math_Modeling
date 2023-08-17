import seaborn as sns
import pandas as pd
df = pd.read_csv("C:\\Users\\xiaohong\\Desktop\\数据项目组\\numbers_OCR_project.csv")

iris = sns.load_dataset()
species = iris.pop("species")
sns.clustermap(iris)
sns.clustermap(
    iris,
    figsize=(7, 5),
    row_cluster=False,
    dendrogram_ratio=(.1, .2),
    cbar_pos=(0, .2, .03, .4)
)