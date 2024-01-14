#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
RNN实现多分类任务
a出现在第几个位置
"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim,hidden_size, sentence_length, vocab):
        # vector_dim 输入维度
        # sentence_length 输入序列长度
        # hidden_size 隐藏维度
        # vocab 字符表
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)  #embedding层
        self.pool = nn.AvgPool1d(sentence_length)   #池化层
        self.layer = nn.RNN(vector_dim, hidden_size, bias=False, batch_first=True)
        self.classify = nn.Linear(vector_dim, 6)     #线性层
        self.activation = torch.sigmoid     #sigmoid归一化函数
        self.loss = nn.functional.mse_loss  #loss函数采用均方差损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)                      #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x = x.transpose(1, 2)                      #(batch_size, sen_len, vector_dim) -> (batch_size, vector_dim, sen_len)
        x = self.pool(x)                           #(batch_size, vector_dim, sen_len)->(batch_size, vector_dim, 1)
        x = x.squeeze()                            #(batch_size, vector_dim, 1) -> (batch_size, vector_dim)
        x = self.classify(x)                       #(batch_size, vector_dim) -> (batch_size, 1) 3*5 5*1 -> 3*1
        y_pred = self.activation(x)                #(batch_size, 1) -> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)   #预测值和真实值计算损失
        else:
            return y_pred                 #输出预测结果

def build_vocab():
    chars = "abcdefghijk"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab)
    return vocab

#随机生成一个样本
def build_sample(vocab, sentence_length):
    #随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]

    # 指定"a"字符出现时为正样本，并返回位置信息
    if 'a' in x:
        # 判断'a'在第几个位置，假设'a'在第一个位置，返回1，第二个位置返回2，以此类推
        y = x.index('a') + 1
        print("a的位置：" + str(y))
    else:
        y = sentence_length
        print("sentence_length：" + str(y))
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
    return x, y

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim, hidden_size, sentence_length):
    model = TorchModel(char_dim, hidden_size, sentence_length, vocab)
    return model

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(100, vocab, sample_length)
    print("本次预测集中共有%d个正样本，%d个负样本"%(sum(y), 100 - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      # 使用模型对输入数据 x 进行预测
        # 循环遍历预测标签 y_p 与 真实标签 y_t 比较是否一致
        for y_p, y_t in zip(y_pred, y):
            if torch.argmax(y_p) == int(y_t):
                correct += 1   #负样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


def main():
    #配置参数
    epoch_num = 60        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 500    #每轮训练总共训练的样本总数
    hidden_size = 40      # 隐藏维度
    char_dim = 40        #每个字的维度
    sentence_length = 6   #样本文本长度
    learning_rate = 0.0001 #学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, hidden_size, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length) #构造一组训练样本
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)   #测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    #画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  #画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  #画loss曲线
    plt.legend()
    plt.show()
    #保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 40  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    hidden_size = 40
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(vocab, char_dim, hidden_size, sentence_length)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    # 序列化输入字符串
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])
    model.eval()   #测试模式

        # 进行模型预测
    with torch.no_grad():
        result = model(torch.LongTensor(x))  # 注意：这里应该是 model(x) 而不是 model.forward(x)

    # 打印预测结果
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (input_string, torch.argmax(result[i]), result[i])) #打印结果


if __name__ == "__main__":
    main()
    test_strings = ["fadghc", "kdadfg", "ecadeg", "hakckc","adghke"]
    predict("model.pth", "vocab.json", test_strings)
