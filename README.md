## 简介

使用tensorflow中的rnn相关操作，以《全宋词》为训练数据，训练一个人工智能写词机。

### word embedding 部分

参考https://www.tensorflow.org/tutorials/word2vec的内容，以下述脚本为基础，完成对提供的《全宋词》的embedding.

https://www.github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/word2vec/word2vec_basic.py

下图为在embedding脚本运行完之后输出。

![embedding](tsne.png)

图片中意义接近的词，如数字等，距离比较近（一这个数字是个特例，离其他数字比较远）。

提供一个文档，说明自己对embedding的理解，以及对上述图片的结果分析和认识。

#### 要点提示

- 全宋词资料不同于英文，不使用分词，这里直接将每个单字符作为一个word。
- 全宋词全文共6010个不同的单字符，这里只取出现次数最多的前5000个单字符。
- 后面的RNN训练部分，需要使用embedding的dictionary, reversed_dictionary,请使用json模块的save方法将这里生成的两个字典保存起来。utils中也提供了一个字典的生成方法，RNN部分，如果不使用生成的embedding.npy文件作为model的embeding参数（参考model的build方法中的embedding_file参数）的时候可以使用这个utils中提供的方法直接生成这两个字典文件。
- matplotlib中输出中文的时候会出现乱码，请自行搜索如何设置matplotlib使之可以输出中文。
- 按照tensorflow官方代码中给出的设置，运行40W个step可以输出一个比较好的结果，四核CPU上两三个小时左右。
- 对于文本的处理，可以搜到很多不同的处理方式，大部分文本处理都要删掉所有的空格，换行，标点符号等等。这里的训练可以不对文本做任何处理。
- 本项目涉及大量中文的处理，因为python2本身对UTF-8支持不好，另外官方对python2的支持已经快要结束了，推荐本项目使用python3进行。

>
```py
# word2vec中，可以使用如下代码来保存最终生成的embeding
np.save('embedding.npy', final_embeddings)
```

### rnn训练部分

需要实现RNN网络部分,RNN数据处理部分和RNN训练部分。
- train.py 训练
- utils.py 数据处理
- model.py 网络

训练的输出log输出中可以看到下述内容

```sh
2018-01--- --:--:-,114 - DEBUG - sample.py:77 - ==============[江神子]==============
2018-01--- --:--:-,114 - DEBUG - sample.py:78 - 江神子寿韵）

一里春风，一里春风，一里春风，一里春风，不是春风。

一里春风，不是春风，不是春风。不是春风，不是春风。

浣溪沙（春
2018-01--- --:--:-,556 - DEBUG - sample.py:77 - ==============[蝶恋花]==============
2018-01--- --:--:-,557 - DEBUG - sample.py:78 - 蝶恋花寿韵）

春风不处。一里春风，一里春风，不是春风。不是春风，不是春风，不是春风。

一里春风，不是春风，不是春风。不是春风，不是
2018-01--- --:--:-,938 - DEBUG - sample.py:77 - ==============[渔家傲]==============
2018-01--- --:--:-,940 - DEBUG - sample.py:78 - 渔家傲
一里春风，一里春风，一里春风，一里春风，不是春风。

水调歌头（寿韵）

春风不处，一里春风，一里春风，一里春风，不是春风。
```

可以明确看到，RNN学会了标点的使用，记住了一些词牌的名字。

#### 要点提示

- 构建RNN网络需要的API如下，请自行查找tensorflow相关文档。
    - tf.nn.rnn_cell.DropoutWrapper
    - tf.nn.rnn_cell.BasicLSTMCell
    - tf.nn.rnn_cell.MultiRNNCell
- RNN部分直接以embedding作为输入，所以其hiddenunit这里取128,也就是embedding的维度即可。
- RNN的输出是维度128的，是个batch_size*num_steps*128这种的输出，为了做loss方便，对输出进行了一些处理，concat，flatten等。具体请参考api文档和代码。
- RNN输出的维度与num_words维度不符，所以需要在最后再加一个矩阵乘法，用一个128*num_words的矩阵将输出维度转换为num_words。
- RNN可能出现梯度爆炸或者消失的问题，对于梯度爆炸，这里直接对gradient做了裁剪，细节参考model代码。
- 这里模型的规模比较小，所以输出的内容可能不是特别有意义，而且训练过程中，不同的checkpoint，其输出也有一些区别。
- 数据处理中，data为文本中一段随机截取的文字，label为data对应的下一个标号的文字。以苏轼的江神子（江城子）为例：输入为 “老夫聊发少年”，则对应的label为"夫聊发少年狂"。
- 训练过程至少要到第二个epoch才能看到一些比较有意义的输出，第一个epoch的输出可能是大量的标点，换行等等。而且这种情况后面还会有。
- 这里的代码，train_eval.py用于在tinymind上运行训练和采样，按照代码中默认的设置，运行一个epoch需要19220步，在tinymind上需要半小时左右。
- rnn中，dictiionary和reverse_dictionary为汉字的索引，可以使用word embeding生成的，也可以重新生成这两个字典。如果model.build中使用word embeding中生成的embeding_file.npy，则为了保证汉字索引的对应关系，必须使用与embeding_file.npy一起生成的dictionary和reverse_dictionary

## 参考资料

各文件简介：
- flags.py 命令行参数处理
- model.py 模型定义
- QuanSongCi.txt 《全宋词》文本
- sample.py 用最近的checkpoint，对三个词牌进行生成操作，结果似乎不是很好
- train.py 训练脚本
- utils.py 数据读取，训练数据生成等
