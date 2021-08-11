### A complete pytorch implementation of skipgram model (with subsampling and negative sampling)

对word2vec skip-gram negative sample的复现。


### RUN

 - 整理数据集格式，形如`dataset/data.txt`(这里使用的以空格键作为分词符,可以参考`propress.py`处理文本)
 - 配置自己的`config.py`(主要是各种路径)
 - 运行`python word2vec.py`进行word2vec模型训练

### 阅读代码
 - 主要参考为 tmikolov的`word2vec.c`实现, 同时也参考了一些其他Python版本的实现
 - 要了解word2vec的实现, 阅读`dataset.py`和`skipgram.py`两个文件即可

### Trick
 得益于PyTorch的自动微分,故并未实现原实现中对sigmoid函数的近似处理(分块映射)
 - Skip-gram
 - Negative Sampling
 - Sub-sampling
 - Dynamic Window


