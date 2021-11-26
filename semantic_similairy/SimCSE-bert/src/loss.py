'''
定义 simple contrastive sentence embedding 的loss function
'''


import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
simCSE_loss 实现细节：
    （1）pred 生成方法，[a, b, c, d] -> [a, a, b, b, c, c, d, d]
    （2）label 的的生成方法：label = ids + 1 - ids % 2 * 2
    （3）屏蔽对象本身similarity的方式：减去一个很大得知，使得计算指数时达到被屏蔽的效果。
'''



def simCSE_loss(pred, tau=0.05):
    ids = torch.arange(pred.shape[0], device=device)
    y_true = ids + 1 - ids % 2 * 2
    #print(y_true)
    similarity = F.cosine_similarity(pred.unsqueeze(1), pred.unsqueeze(0), dim=2)
    # 将对角线部分 也就是自身相等部分 进行屏蔽 给一个很大的负数，在指数计算过程中基本为0
    similarity = similarity - torch.eye(pred.shape[0], device=device) * 1e12
    # similarity除以 tau 进行放大
    similarity = similarity / tau
    loss = F.cross_entropy(similarity, y_true)
    return loss


if __name__ == '__main__':
    pred = torch.tensor([[0.3, 0.2, 2.1, 3.1],
        [0.3, 0.2, 2.1, 3.1],
        [-1.79, -3, 2.11, 0.89],
        [-1.79, -3, 2.11, 0.89]])
    loss = simCSE_loss(pred)