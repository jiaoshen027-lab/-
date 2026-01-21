# - Word2vec论文赏析
## Efficient Estimation of Word Representations in Vector Space

---

### **摘要**
- 提出了两种新颖的神经网络模型架构，用于从大规模数据集中学习高质量的词向量表示。
- 这些词向量在语法和语义任务中表现出色，并且计算成本显著低于之前的方法。
- 模型能够在不到一天的时间内从数十亿词的语料中学习到高质量的词向量。

### **引言**
- 传统 NLP 方法将词视为离散符号，但分布式词向量能够捕捉词之间的相似性和复杂关系。
- 本文目标是构建高效模型，从数十亿级语料中学习高质量词向量，并研究其线性关系（如 “King - Man + Woman ≈ Queen”）。
- 之前的方法计算复杂度高，难以扩展到大规模数据，本文旨在提出更高效的替代方案。

### **经典模型架构及优化**
- 介绍了前馈神经网络语言模型（NNLM）和循环神经网络语言模型（RNNLM）的结构及其计算复杂度，这两种模型作为基线架构。
- 为应对输出层计算挑战，使用层次 Softmax 和 Huffman 编码来减少输出层的计算开销，提升训练效率，将输出计算复杂度从O(V)降至O(log(V))。
- 提升训练速度：提出了基于分布式框架 DistBelief 的并行训练方法，支持多副本异步梯度下降。

### **新的对数线性模型**
- 提出了两种简化模型：连续词袋模型（CBOW）和 Skip-gram 模型，移除了隐藏层以降低计算复杂度。
- CBOW 通过上下文预测当前词，Skip-gram 通过当前词预测上下文，两者均使用词向量的平均或投影。
- 这两个模型可以在大规模语料上高效训练，且能保持词之间的线性语义关系。

### **结果**
- 构建了一个包含语法和语义关系的综合测试集，用于评估词向量的质量。
- 实验表明，CBOW 和 Skip-gram 在语法和语义任务上优于之前的 NNLM 和 RNNLM，且训练速度更快。
- 在大规模并行训练中，模型进一步提升了表现，尤其在语义任务上 Skip-gram 表现最佳。

### **学习到的关系示例**
- 展示了词向量能够捕捉多种语义和语法关系，如国家-首都、形容词-副词、词语形态变化等。
- 通过向量加减法（如 “Paris - France + Italy ≈ Rome”）可以推理出未见过的关系。
- 结果表明，更大维度和更多数据能进一步提升关系的准确性和泛化能力。

### **结论**
- 本文表明，简单的模型（CBOW 和 Skip-gram）能够高效学习高质量词向量，适用于大规模语料。
- 这些词向量在多类 NLP 任务中表现出色，有望成为未来应用的基础组件。
- 未来工作包括扩展到更大语料、融入词形态信息，以及在知识图谱、机器翻译等任务中的应用。

---
### 理解
- word2vec是创造词嵌入（将一个词表示为一个列向量，其他的词嵌入方法：fastText, GloVe, ELMO, BERT, GPT-2）的一种方法
- 类似于完型填空，空格前N个单词和空格后N个单词已知，推测出空格单词，这里的N是超参数，如果N变大，可以更好地理解空格单词，但是计算开销会变大（N取4或5）
- 两层：嵌入层（输出为300维）和线性层后接softmax激活，相当于做了一个多分类
- 词袋模型（CBOW）对应于完型填空，由上下文预测空格，输入上下文单词，每个单词进入同样的嵌入层（并行），结果取平均，进入线性层，最后进入softmax激活
- Skip-gram模型由当前词预测上下文，输入当前词，进入嵌入层后得到的结果直接进入线性层，最后进入softmax
- 我们并不关注结果本身，而是要它的副产品——词向量
- 无监督学习，需要大量语料，最初的语料库包含了 60 亿个tokens
- 在训练的时候要注意语料和我们想要的结果对应，比如做新闻摘要提取的时候，就要新闻的语料....
- 创建词表，一个token对应一个数字，token在语料中常常出现，如果出现次数过少，就不将其加入此表；注意，标点符号和其他特殊符号也作为独立的token加入词表中

---
### Code（Pytorch）
#### model
```python
import torch.nn as nn
from utils.constants import EMBED_DIMENSION, EMBED_MAX_NORM

class CBOW_Model(nn.Module):
    def __init__(self, vocab_size: int):
        super(CBOW_Model, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIMENSION,
            max_norm=EMBED_MAX_NORM,
        )
        self.linear = nn.Linear(
            in_features=EMBED_DIMENSION,
            out_features=vocab_size,
        )

    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x


class SkipGram_Model(nn.Module):
    def __init__(self, vocab_size: int):
        super(SkipGram_Model, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIMENSION,
            max_norm=EMBED_MAX_NORM,
        )
        self.linear = nn.Linear(
            in_features=EMBED_DIMENSION,
            out_features=vocab_size,
        )

    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = self.linear(x)
        return x
```
#### 代码说明
- 首先随机初始化词表矩阵（n*300），假如某个token的ID为i，输入i，输出对应的第i行就是该token的词向量
- 上面代码没有softmax层，因为Pytorch中的CrossEntropyLoss接收的就是未经归一化处理的原始分数，相当于已经内置了Softmax，我现在的理解是CrossEntropyLoss是softmax的一种对数表达
- EMBED_MAX_NORM要求对每一行向量的范数小于等于1，相当于做了一个截断，如果大于1，则将它的结果限制为1，如果本身范数小于1，则不必理会，可以避免梯度爆炸的同时有效防止过拟合，此外，这样还有一个好处，相比于不做处理的，它可以让相似的单词对应的词向量之间的cos也很相近
  - 我的问题是，为什么不限制都等于1呢
  - 可以保持多样性呀，专业词可以倾向于对应较大的范数，高频词（比如你我他）范数通常比较小，保留距离原点的距离信息（向量范数本质上编码了模型对该词表示的"置信度"或"重要性"）
- 将所有文章出现次数较少的token，词表中不存在的词统一ID，比如设为0

---

#### 常量设置
```python
CBOW_N_WORDS = 4
SKIPGRAM_N_WORDS = 4

MIN_WORD_FREQUENCY = 50
MAX_SEQUENCE_LENGTH = 256

EMBED_DIMENSION = 300
EMBED_MAX_NORM = 1
```

----
#### dataloader
```python
import torch
from functools import partial
from torch.utils.data import DataLoader
from torchtext.data import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import WikiText2, WikiText103

from utils.constants import (
    CBOW_N_WORDS,
    SKIPGRAM_N_WORDS,
    MIN_WORD_FREQUENCY,
    MAX_SEQUENCE_LENGTH,
)

def get_english_tokenizer():
    """
    Documentation:
    https://pytorch.org/text/stable/_modules/torchtext/data/utils.html#get_tokenizer
    """
    tokenizer = get_tokenizer("basic_english", language="en")
    return tokenizer

def get_data_iterator(ds_name, ds_type, data_dir):
    if ds_name == "WikiText2":
        data_iter = WikiText2(root=data_dir, split=(ds_type))
    elif ds_name == "WikiText103":
        data_iter = WikiText103(root=data_dir, split=(ds_type))
    else:
        raise ValueError("Choose dataset from: WikiText2, WikiText103")
    data_iter = to_map_style_dataset(data_iter)
    return data_iter

def build_vocab(data_iter, tokenizer):
    """Builds vocabulary from iterator"""
    
    vocab = build_vocab_from_iterator(
        map(tokenizer, data_iter),
        specials=["<unk>"],
        min_freq=MIN_WORD_FREQUENCY,
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab

def collate_cbow(batch, text_pipeline):
    """
    Collate_fn for CBOW model to be used with Dataloader.
    `batch` is expected to be list of text paragrahs.
    
    Context is represented as N=CBOW_N_WORDS past words 
    and N=CBOW_N_WORDS future words.
    
    Long paragraphs will be truncated to contain
    no more that MAX_SEQUENCE_LENGTH tokens.
    
    Each element in `batch_input` is N=CBOW_N_WORDS*2 context words.
    Each element in `batch_output` is a middle word.
    """
    batch_input, batch_output = [], []
    for text in batch:
        text_tokens_ids = text_pipeline(text)

        if len(text_tokens_ids) < CBOW_N_WORDS * 2 + 1:
            continue

        if MAX_SEQUENCE_LENGTH:
            text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]

        for idx in range(len(text_tokens_ids) - CBOW_N_WORDS * 2):
            token_id_sequence = text_tokens_ids[idx : (idx + CBOW_N_WORDS * 2 + 1)]
            output = token_id_sequence.pop(CBOW_N_WORDS)
            input_ = token_id_sequence
            batch_input.append(input_)
            batch_output.append(output)

    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output


def collate_skipgram(batch, text_pipeline):
    """
    Collate_fn for Skip-Gram model to be used with Dataloader.
    `batch` is expected to be list of text paragrahs.
    
    Context is represented as N=SKIPGRAM_N_WORDS past words 
    and N=SKIPGRAM_N_WORDS future words.
    
    Long paragraphs will be truncated to contain
    no more that MAX_SEQUENCE_LENGTH tokens.
    
    Each element in `batch_input` is a middle word.
    Each element in `batch_output` is a context word.
    """
    batch_input, batch_output = [], []
    for text in batch:
        text_tokens_ids = text_pipeline(text)

        if len(text_tokens_ids) < SKIPGRAM_N_WORDS * 2 + 1:
            continue

        if MAX_SEQUENCE_LENGTH:
            text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]

        for idx in range(len(text_tokens_ids) - SKIPGRAM_N_WORDS * 2):
            token_id_sequence = text_tokens_ids[idx : (idx + SKIPGRAM_N_WORDS * 2 + 1)]
            input_ = token_id_sequence.pop(SKIPGRAM_N_WORDS)
            outputs = token_id_sequence

            for output in outputs:
                batch_input.append(input_)
                batch_output.append(output)

    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output

def get_dataloader_and_vocab(
    model_name, ds_name, ds_type, data_dir, batch_size, shuffle, vocab=None
):

    data_iter = get_data_iterator(ds_name, ds_type, data_dir)
    tokenizer = get_english_tokenizer()

    if not vocab:
        vocab = build_vocab(data_iter, tokenizer)
        
    text_pipeline = lambda x: vocab(tokenizer(x))

    if model_name == "cbow":
        collate_fn = collate_cbow
    elif model_name == "skipgram":
        collate_fn = collate_skipgram
    else:
        raise ValueError("Choose model from: cbow, skipgram")

    dataloader = DataLoader(
        data_iter,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_fn, text_pipeline=text_pipeline),
    )
    return dataloader, vocab
```
#### dataloader代码说明
- 对于每一个文章段落：
  - 转为小写，分词，用ID进行编码
  - 段落太短，跳过；段落太长，截断
  - 使用大小为9的滑动窗口（4个前文单词，中间词，4个后文单词）遍历该段落，这里是整段处理，即将一段话看成一个句子。
  - 将所有中间词合并为一个列表，将他们作为Y
  - 将所用上下文单词合并为一个列表的列表，记为X
  - 将所有段落的X和所有段落的Y合并在一起，成为批量的X和Y
  - 注意，当我们调用 collate_fn 时，最终生成的批次（Xs 和 Ys）数量将与 DataLoader 中指定的 batch_size 参数不同，并且会因段落的不同而变化。生成的批次是迭代次数，这里重点理解一下
- batch size:一次性输入模型进行前向/反向传播的样本数量
- iteration:完成一次梯度更新的训练步骤
- epoch:完整遍历整个训练数据集一次的训练过程
- 举个例子，假如有1000个段落，这里要求打乱了，设置每次batch_size为32，则每次取32个段落，如果不考虑因段落太短省去的段落，则iteration为1000/32向上取整为32次，而将这32次训练结束，则完成了一个epoch

---
#### train
```python
import os
import numpy as np
import json
import torch


class Trainer:
    """Main class for model training"""
    
    def __init__(
        self,
        model,
        epochs,
        train_dataloader,
        train_steps,
        val_dataloader,
        val_steps,
        checkpoint_frequency,
        criterion,
        optimizer,
        lr_scheduler,
        device,
        model_dir,
        model_name,
    ):  
        self.model = model
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.train_steps = train_steps
        self.val_dataloader = val_dataloader
        self.val_steps = val_steps
        self.criterion = criterion
        self.optimizer = optimizer
        self.checkpoint_frequency = checkpoint_frequency
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.model_dir = model_dir
        self.model_name = model_name

        self.loss = {"train": [], "val": []}
        self.model.to(self.device)

    def train(self):
        for epoch in range(self.epochs):
            self._train_epoch()
            self._validate_epoch()
            print(
                "Epoch: {}/{}, Train Loss={:.5f}, Val Loss={:.5f}".format(
                    epoch + 1,
                    self.epochs,
                    self.loss["train"][-1],
                    self.loss["val"][-1],
                )
            )

            self.lr_scheduler.step()

            if self.checkpoint_frequency:
                self._save_checkpoint(epoch)

    def _train_epoch(self):
        self.model.train()
        running_loss = []

        for i, batch_data in enumerate(self.train_dataloader, 1):
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss.append(loss.item())

            if i == self.train_steps:
                break

        epoch_loss = np.mean(running_loss)
        self.loss["train"].append(epoch_loss)

    def _validate_epoch(self):
        self.model.eval()
        running_loss = []

        with torch.no_grad():
            for i, batch_data in enumerate(self.val_dataloader, 1):
                inputs = batch_data[0].to(self.device)
                labels = batch_data[1].to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss.append(loss.item())

                if i == self.val_steps:
                    break

        epoch_loss = np.mean(running_loss)
        self.loss["val"].append(epoch_loss)

    def _save_checkpoint(self, epoch):
        """Save model checkpoint to `self.model_dir` directory"""
        epoch_num = epoch + 1
        if epoch_num % self.checkpoint_frequency == 0:
            model_path = "checkpoint_{}.pt".format(str(epoch_num).zfill(3))
            model_path = os.path.join(self.model_dir, model_path)
            torch.save(self.model, model_path)

    def save_model(self):
        """Save final model to `self.model_dir` directory"""
        model_path = os.path.join(self.model_dir, "model.pt")
        torch.save(self.model, model_path)

    def save_loss(self):
        """Save train/val loss as json file to `self.model_dir` directory"""
        loss_path = os.path.join(self.model_dir, "loss.json")
        with open(loss_path, "w") as fp:
            json.dump(self.loss, fp)
```

---
#### helper
```python
import os
import yaml
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from utils.model import CBOW_Model, SkipGram_Model


def get_model_class(model_name: str):
    if model_name == "cbow":
        return CBOW_Model
    elif model_name == "skipgram":
        return SkipGram_Model
    else:
        raise ValueError("Choose model_name from: cbow, skipgram")
        return


def get_optimizer_class(name: str):
    if name == "Adam":
        return optim.Adam
    else:
        raise ValueError("Choose optimizer from: Adam")
        return
    

def get_lr_scheduler(optimizer, total_epochs: int, verbose: bool = True):
    """
    Scheduler to linearly decrease learning rate, 
    so thatlearning rate after the last epoch is 0.
    """
    lr_lambda = lambda epoch: (total_epochs - epoch) / total_epochs
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=verbose)
    return lr_scheduler


def save_config(config: dict, model_dir: str):
    """Save config file to `model_dir` directory"""
    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path, "w") as stream:
        yaml.dump(config, stream)
        
        
def save_vocab(vocab, model_dir: str):
    """Save vocab file to `model_dir` directory"""
    vocab_path = os.path.join(model_dir, "vocab.pt")
    torch.save(vocab, vocab_path)
    
```

