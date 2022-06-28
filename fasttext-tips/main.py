import fasttext


def train(path_train, model_output):
    model = fasttext.train_supervised(input=path_train, lr=0.5, epoch=120, wordNgrams=2, bucket=200000, dim=50, loss='ova')
    model.save_model(model_output)
    print('finish save model')

def auto_hyperparameter_optimization(path_train, path_test, model_output):
    model = fasttext.train_supervised(input=path_train,
                                      autotuneValidationFile=path_test,
                                      autotuneDuration=600)
    model.save_model(model_output)

def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

def valid(path_test, model_output):
    model = fasttext.load_model(model_output)
    print_results(*model.test(path_test, k=5))


if __name__ == '__main__':
    path_train = 'data/bot_corpus.train_v2'
    path_test = 'data/bot_corpus.test_v2'
    model_output = 'output/model_bot_v1.bin'
    model_output2 = 'output/model_bot_v2.bin'
    model_output3 = 'output/model_bot_v3.bin'
    model_output4 = 'output/model_bot_v4.bin'
    #train(path_train, model_output)
    auto_hyperparameter_optimization(path_train, path_test, model_output4)
    valid(path_test, model_output4)
