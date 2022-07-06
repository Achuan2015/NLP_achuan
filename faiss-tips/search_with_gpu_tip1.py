'''
reference：https://zhuanlan.zhihu.com/p/107241260
'''


import faiss

# 创建索引
d= 1024
index = faiss.index_factory(d, "IVF1024,SQ8")

## transfer 到GPU设备上
gpu_id = 0
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_all_gpu(res, gpu_id, index)

# 训练模型

# 搜索

#保存模型模型前，需要首先transfer到cpu上
index_cpu = faiss.index_gpu_to_cpu(gpu_index)
faiss.write_index(index_cpu, 'IVF1024_SQ8.index')

#加载模型
new_index = faiss.read_index('IVF1024_SQ8.index')

#如果需要在gpu上使用，需要再次transfer到GPU设备上
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, index)