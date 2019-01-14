
**文本分类之 Rnn via Attention**

文本分类任务我在前几章中已经介绍了两个模型，分别是TFIDF+LR和TFIDF+NBSVM。这两个是都是利用TFIDF提取词频以及ngram信息作为特征，然后利用传统的机器学习方法作为分类器。随着深度学习的发展，基于词频的方法表征文本特征始终是基于标量的形式表征，然而不能够深度的表征词语的信息，因此，word2vec,glove以及fasttext等基于向量的表征方法可以有效的缓解词语表征能力的不足。同时，更随着硬件的发展，基于语境的文本向量表征方法如：ULMFiT,ELMO,BERT等奔涌而现。

在本文中，我将借助有毒评论分类比赛数据https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge， 介绍一种Rnn via Attention 算法作为深度文本分类算法的baseline,帮助有需要的朋友入门深度自然语言处理。

1、导入后面文章中需要的算法库，本文主要是基于keras实现的(一个高度集成的API文档，后端使用Tenforflow)。


```python
import sys, os, re, csv, codecs, numpy as np, pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras import backend as K
from keras.engine.topology import Layer
```

    Using TensorFlow backend.


2、读取我们的数据并替换缺失的值，如果你想要得到很好的效果，可以对这些评论数据进行预处理，比如单词缩写，复数，等等。


```python

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

list_sentences_train = train["comment_text"].fillna("_na_").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("_na_").values
```

3、需要对数据集中的评论数据进行向量化。在向量化之前呢，首先需要对这些单词进行编码，也就是将单词转化为数字，因为计算机只能只能识别数字；其实在keras中有个专门的函数，可以帮助我们简单的将单词进行编码。另外，每个句子的长度长短不一，需要选择一个合适的长度，比如100长度的句子可以覆盖90%的文本。在本文中，我们选择句子长度为100，保留文中频率在前20000的单词。


```python

embed_size = 50 #单词向量维度
max_features = 20000 #最多使用多少词，换句话说，也就是我们保留频率前多少词进行embedding，至于之后的我们就忽略掉。
maxlen = 100 #句子选取的长度
#对于以上的三个参数，需要我们对具体的文本进行统计分析。

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train)+list(list_sentences_test))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
```


```python
X_t.shape,X_te.shape
```




    ((159571, 100), (153164, 100))



4、将单词转化成向量，可以用的方法有很多，word2vec,fasttext,glove等，在本文中，我们用了预训练维度为50的词向量，斯坦福提供的预训练的英文词向量，大家可以在这个链接上下载：http://nlp.stanford.edu/data/glove.6B.zip 。 当然，你也可以自己训练词向量，这里我就不在介绍了。斯坦福也提供了不同的词向量，大家可以去这个官网下载： https://nlp.stanford.edu/projects/glove/ 。

具体地，对于本文来说，我们首先随机初始化一个标准正太矩阵，这样做的有一个好处是，对于有毒评论文本数据的单词在预训练中不存在，我们可以使用一个随机标准正太分布的向量代替它。然后对于存在的单词，我们就使用的预训练的词向量。代码实现的话，大家可以参考下面的，这些代码也可以迁移到其他的任务中。


```python
EMBEDDING_FILE = "../input/glove.6B.50d.txt"
def get_coefs(word,*arr): 
    return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: 
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: 
        embedding_matrix[i] = embedding_vector
embedding_matrix.shape      
```




    (20000, 50)



5、下面就是本文的重点了，本文使用Rnn via Attention 模型,一个常用解决文本分类任务的baseline。对于文本数据，最重要的是如何捕捉到上下文信息。就像一个人一样，如果他讲话虎头蛇尾，就很难捕捉到有用的信息，别人很难理解；相反，如果一个人讲话有条有理，别人一听就很通俗易懂。同样地，计算机也是一样，如果嵌入一个有条有理的算法，计算机很快就能读懂重要的信息。RNN主要解决序列数据的处理，比如文本、语音、视频等等。简单的来说，RNN主要是通过上一时刻的信息以及当前时刻的输入，确定当前时刻的信息。因此，RNN可以捕捉到序列信息，这与捕捉文本的上下文信息相得益彰。

传统的RNN也会存在许多问题，无法处理随着递归，权重指数级爆炸或消失的问题（Vanishing gradient problem），难以捕捉长期时间关联等等。基于RNN的变体，如LSTM和GRU很好的缓解这个问题。但是呢，LSTM和GRU这些网络，在长序列信息记忆方面效果也不是很好，Colin Raffel等人基于RNN以及RNN的变体，提出了一种适用于RNN(s)的简化注意力模型，很好的解决序列长度的合成加法和乘法的长期记忆问题。

在本文中，我们使用了一种RNN的变体——LSTM捕捉文本的上下文信息，使用Colin Raffel等人提出的attention机制代替池化层，作为我们的模型。更多地，人类在阅读文本时会考虑到前后上下文信息，在本文中，我们使用了双向的LSTM来捕捉前后上下文信息，充分的捕捉文本的前后信息。

(1)如下图所示，下图是一个双向的LSTM模型结构示意图。



```python

```

(2)下面的基于keras实现的attention, attention的结构示意图如下所示：


```python

```


```python
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)
        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim
```

(3)下面我们就可以实现我们的rnn-attention模型了，首先通过embedding函数，将我们的单词id转化成(None,100,50)维度的向量,注意"None"代表我们的batch_size;然后利用双向的LSTM捕捉文本的上下文信息，得到一个(None,100,100)维度的向量；接着利用attention机制捕捉文本中最关键的信息，一个(None,100)维度的向量，至于attention的可视化，大家可以参考论文；最后就是接入一个50维的FC层；由于评价标准是每个类别auc的平均值，我们激活函数选择sigmoid,最后得到每个类别的概率。具体的实现可以参考下面的代码，损失函数选择binary_crossentropy(交叉熵代价函数)；优化函数选择adam。


```python
def rnn_attention(maxlen):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
    x = Attention(step_dim=100)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model
model=rnn_attention(maxlen)
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         (None, 100)               0         
    _________________________________________________________________
    embedding_1 (Embedding)      (None, 100, 50)           1000000   
    _________________________________________________________________
    bidirectional_1 (Bidirection (None, 100, 100)          40400     
    _________________________________________________________________
    attention_1 (Attention)      (None, 100)               200       
    _________________________________________________________________
    dense_1 (Dense)              (None, 50)                5050      
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 50)                0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 6)                 306       
    =================================================================
    Total params: 1,045,956
    Trainable params: 1,045,956
    Non-trainable params: 0
    _________________________________________________________________


(4)现在我们的模型构建完成了，开始训练我们的模型，我们随机切取10%的样本作为我们模型的验证集。


```python
model.fit(X_t, y, batch_size=32, epochs=2, validation_split=0.1)
```

    Train on 143613 samples, validate on 15958 samples
    Epoch 1/2
    143613/143613 [==============================] - 912s 6ms/step - loss: 0.0658 - acc: 0.9779 - val_loss: 0.0518 - val_acc: 0.9814
    Epoch 2/2
    143613/143613 [==============================] - 895s 6ms/step - loss: 0.0467 - acc: 0.9825 - val_loss: 0.0476 - val_acc: 0.9822





    <keras.callbacks.History at 0x7fe2445ce8d0>




```python
submission = pd.DataFrame.from_dict({'id': test['id']})
y_test = model.predict([X_te], batch_size=1024, verbose=1)
submission[list_classes] = pd.DataFrame(y_test)
```


```python
submission.to_csv("rnn_attention.csv",index=False)
```

将其结果提交到线上的评测网站，线上提交的结果是分数是0.9848.

总结
在这篇文章，我主要通过有毒评论分类数据集简单介绍了Rnn via Attention 模型，Rnn via Attention 模型常作于文本分类的baseline，大家也可以将Rnn via Attention 应用于其他的文本分类算法，如 情感分析等。

参考

attention机制的实现：https://www.kaggle.com/takuok/bidirectional-lstm-and-attention-lb-0-043

模型参考：https://www.kaggle.com/jhoward/improved-lstm-baseline-glove-dropout

attention论文：Raffel C, Ellis D P W. Feed-forward networks with attention can solve some long-term memory problems[J]. arXiv preprint arXiv:1512.08756, 2015.


```python

```
