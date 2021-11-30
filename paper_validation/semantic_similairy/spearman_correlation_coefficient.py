'''
熟悉一下spearman correlation coefficient 的应用方法，这个指标可以用来精排 而且还不用那么在意阈值

参考链接：[三大相关系数: pearson, spearman, kendall（python示例实现）](]https://www.cnblogs.com/yjd_hycf_space/p/11537153.html)
'''

import pandas as pd
import numpy as np

# 原始数据
X1=pd.Series([1, 2, 3, 4, 5, 6])
Y1=pd.Series([0.3, 0.9, 2.7, 2, 3.5, 5])

# 处理数据删除Nan
x1=X1.dropna()
y1=Y1.dropna()

n=x1.count()
print('n', n)

x1.index=np.arange(n)
y1.index=np.arange(n)


# 分部计算
dist = (x1.sort_values().index - y1.sort_values().index) ** 2
dist_norm = dist.to_series().sum()

p = 1 - n * dist_norm / (n * (n ** 2 - 1))

print('自定义p', p)

# 使用pandas 的接口计算
r = x1.corr(y1, method='spearman')
print('r', r)