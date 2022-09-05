'''
一些loss function 的实现合集

reference:
    (1) [final_loss](https://github.com/xiaoboCASIA/SV-X-Softmax/blob/master/loss.py)
    (2) [What is Focal Loss and when should you use it?
](https://amaarora.github.io/2020/06/29/FocalLoss.html)
'''

import torch.nn.functional as F
import torch
import torch.nn as nn


class Focal_Loss(nn.Module):
    '''
     alpha * (1 - f(x)) ** gamma * f(x)
    '''

    def __init__(self, alpha=0.25, gamma=2):
        super(Focal_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        targets = targets.type(torch.long)
        loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return loss.mean()
        


class Contrastive_Loss(nn.Module):

    '''
    contrastive loss function: Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    如果是正样本就是 欧几里得距离，如果是负样本，则要求至少达到 m 的距离（其中m 是常量）
    '''
    
    def __init__(self, margin):
        super(Contrastive_Loss, self).__init__()
        self.margin = margin

    def forward(self, target1, target2, label):
        # 计算欧几里得距离
        euclidean_distance = F.pairwise_distance(target1, target2)
        # 当 label=0 时，loss 的值就是 euclidean distance 的值，如果 label=1 时，loss 的值为 max(0, m - euclidean_distance)
        contrastive_loss =  (1 - label) * torch.pow(euclidean_distance, 2) + label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0), 2)
        return contrastive_loss.mean()