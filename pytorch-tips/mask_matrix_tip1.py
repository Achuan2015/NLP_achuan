"""
分别实现 encoder-decoder， Causal Decoder，Prefix Decoder 的 mask 矩阵
"""

import torch

a1 = torch.randn((3,10))