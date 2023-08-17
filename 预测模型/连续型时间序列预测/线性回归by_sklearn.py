from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split  # 可以生成用于回归分析的样本数据与对应的标签。

X, y, coef = make_regression(n_samples=1000, n_features=2, coef=True, bias=5.5, random_state=0, noise=10)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=0)
print(f"实际权重：{coef}")
lr = LinearRegression()
lr.fit(train_X, train_y)
print(f"权重：{lr.coef_}")
print(f"截距：{lr.intercept_}")
y_hat = lr.predict(test_X)
print(f"均方误差：{mean_squared_error(test_y, y_hat)}")
print(f"训练集R^2：{lr.score(train_X, train_y)}")
print(f"测试集R^2：{lr.score(test_X, test_y)}")
