'''
使用 pytorch 来实现不同的权重 权重初始化方法
- uniform distribution
- gaussian distribution
- zero distribution
'''

import torch
import torch.nn as nn

# 零初始化
class ZeroIintModel(nn.Module):

    def __init__(self):
        super(ZeroIintModel, self).__init__()
        self.fc = nn.Linear(100, 50)
        # 使用零初始化
        nn.init.zeros_(self.fc.weight)

# 高斯初始化
class GaussianInitModel(nn.Module):
    '''
    权重参数初始化为随机的小数值，这些小数值是从高斯分布中采样得到的。这样其实有助于打破对称性，是的每个神经元
    具有不同的初始权重，有助于加速网络的训练过程。
    '''

    def __init__(self):
        super(GaussianInitModel, self).__init__()
        self.fc = nn.Linear(100, 50)
        # 使用高斯初始化
        nn.init.normal_(self.fc.weight, mean=0, std=0.01)  # 从均值为0，标准差为0.01的高斯分布中初始化权重

# xavier unifrom 初始化

class XavierUnifromModel(nn.Module):

    def __init__(self):
        super(XavierUnifromModel, self).__init__()
        self.fc = nn.Linear(100, 50)
        nn.init.xavier_uniform_(self.fc.weight)

if __name__ == "__main__":
    model = ZeroIintModel()
    model = GaussianInitModel()
    model = XavierUnifromModel()
    print(model.fc.weight)