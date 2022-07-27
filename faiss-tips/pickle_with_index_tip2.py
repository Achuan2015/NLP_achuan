from hrtps_file_manager import HRTPSFileManager


def build_index():
    import numpy as np
    import faiss
    d = 64                           # dimension
    nb = 100000                      # database size
    nq = 10000                       # nb of queries
    np.random.seed(1234)             # make reproducible
    xb = np.random.random((nb, d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 1000.
    xq = np.random.random((nq, d)).astype('float32')
    xq[:, 0] += np.arange(nq) / 1000.

    index = faiss.IndexFlatL2(d)   # build the index
    print(index.is_trained)
    index.add(xb)                  # add vectors to the index
    print(index.ntotal)
    return index

def upload_model():
    allow_id = 'zhangc-df04e'
    manager = HRTPSFileManager(allow=allow_id)

    file_key_demo='test_model_01'
    demo_obj = build_index()
    manager.dump(file_key=file_key_demo, obj=demo_obj)

def load_model():
    allow_id = 'zhangc-df04e'
    manager = HRTPSFileManager(allow=allow_id)

    file_key_demo='test_model_01'
    #序列化加载 同 pickle.load
    demo_obj = manager.load(file_key="demo-file-key")
    k = 4                          # we want to see 4 nearest neighbors
    D, I = index.search(xb[:5], k) # sanity check
    print(I)
    print(D)


if __name__ == '__main__':
    pass
