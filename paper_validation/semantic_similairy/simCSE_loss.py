'''
实现simCSE 的unsupervised learning 的loss function

reference link: https://wmathor.com/index.php/archives/1580/
'''


import torch
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def simCSE_loss(pred, tau=0.05):
    ids = torch.arange(pred.shape[0], device=device)
    y_true = ids + 1 - ids % 2 * 2
    cosine_similarity = F.cosine_similarity(pred.unsqueeze(1), pred.unsqueeze(0), dim=2)  # N * N
    cosine_similarity = cosine_similarity - torch.eye(pred.shape[0], device=device) * 1e12
    cosine_similarity = cosine_similarity / tau
    return F.cross_entropy(cosine_similarity, y_true)

loss = simCSE_loss(torch.randn(6, 5))
print('loss', loss)