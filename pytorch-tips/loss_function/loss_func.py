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
     reference:https://amaarora.github.io/2020/06/29/FocalLoss.html
    '''
    def __init__(self, alpha=1, gamma=2):
        super(Focal_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        # （1）使用 F.binary_cross_entropy_with_logits 计算 CE，因此在这里不累计梯度
        #  (2) 得到的 bce_loss = -log(pt)，因此使用指数函数将其还原, torch.exp(-bce_loss) 得到pt
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        targets = targets.type(torch.long)
        # 因为构造的正负样本是采样好的样本，因此不根据正负样本的不同权重采样
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



class Triplet_Loss(nn.Module):

    '''
    思路：
        要使得，anchor 与 positive 样本的距离越来越近，与 negative 的样本越来越远；同时要求 anchor 与 negative 样本的距离
        要比 positive 样本的距离至少要大于 margin的大小

    注意：不同于 contrastive loss， triplet 在训练之前就要根据 label 进行样本的构造
    '''

    def __init__(self, margin):
        super(Triplet_Loss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, target_positive, target_negative):
         p_distance  = F.pairwise_distance(anchor, target_positive)
         n_distance = F.pairwise_distance(anchor, target_negative)
         triplet_loss = torch.pow(torch.clamp(p_distance - n_distance + self.margin, min=0), 2)
         return triplet_loss.mean()