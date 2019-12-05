import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# 加载数据
sns.set(context="notebook", style="whitegrid", palette="dark")
df0 = pd.read_csv('房价预测线性回归/data0.csv', names=['square', 'price'])
sns.lmplot('square', 'price', df0, height=6, fit_reg=True)

df1 = pd.read_csv('房价预测线性回归/data1.csv', names=['square', 'bedrooms', 'price'])
# print(df1.head())

# 绘制3d散点图
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('square')
ax.set_ylabel('bedrooms')
ax.set_zlabel('price')
ax.scatter3D(df1['square'], df1['bedrooms'],
             df1['price'], c=df1['price'], cmap='Greens')

# 数据规范化
def normalize(df):
    return df.apply(lambda col: (col-col.mean())/col.std())

df = normalize(df1)
# print(df.head())

# 绘制规范化数据后的3d散点图
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('square')
ax.set_ylabel('bedrooms')
ax.set_zlabel('price')
ax.scatter3D(df['square'], df['bedrooms'],
             df['price'], c=df['price'], cmap='Reds')
plt.show()

# 添加列
ones = pd.DataFrame({'ones': np.ones(len(df))})
df = pd.concat([ones, df], axis=1)
print(df.head())
