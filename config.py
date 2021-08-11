# encoding:utf-8

configs = {
    'data_path': './dataset/zhihu.txt',
    'model_save_path': './output/model/word2vec.pth',

    'vocab_path': './dataset/processed/vocab.pkl',  # 语料数据
    'pytorch_embedding_path': './output/embedding/pytorch_word2vec2.bin',

    'log_dir': './output/log',
    'figure_dir': './output/figure',
    'stopword_path': './dataset/stopwords.txt'
}

if __name__ == '__main__':
    def print_config(config):
        info = "Running with the following configs:\n"
        for k, v in config.items():
            info += f"\t{k} : {str(v)}\n"
        print("\n" + info + "\n")
        return


    print(print_config(configs))
