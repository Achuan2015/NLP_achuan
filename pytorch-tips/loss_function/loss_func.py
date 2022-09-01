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
        

def focal_loss(pred, target, alpha=0.25, gamma=2):
    '''
     alpha * (1 - f(x)) ** gamma * f(x)
    '''
    bce_loos = F.binary_cross_entropy_with_logits(pred, target, reduction='None')
    