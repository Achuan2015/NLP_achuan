import numpy as np


'''
实现细节:
    (1) 2i 的最大值是 embed_dim 的2倍
    (2) position encoding 会将pos=0的位置预留出来
'''


def get_position_encoding(max_seq_len, embed_dim):
    # 初始化一个position encodding
    # 第一步通过 pos 占据位置
    # PE(pos, 2i) = sim(pos * (1 / 10000^2i/dim)
    # PE(pos, 2i + 1) = cos(pos * ( 1/ 10000^2i+1/dim)
    # 第二步增加 embed_dim -> i 也就是加上 2i -> sim and i -> cos

    ## 第一步 先忽略三角函数 sim 和 cose
    position_encoding = np.array([[pos/np.power(10000, 2 * i/embed_dim) for i in range(embed_dim)] for pos in range(max_seq_len)])
    ## 第一步 加上 三角函数,流出 pos == 0的位置,作为padding的作用
    position_encoding[1:, 0::2] = np.sin(position_encoding[1:, 0::2])     # dim 是 2i 偶数
    position_encoding[1:, 1::2] = np.cos(position_encoding[1:, 1::2])     # dim 是 2i + 1 奇数 
    return position_encoding

position_encoding = get_position_encoding(5, 10)
print(position_encoding)
