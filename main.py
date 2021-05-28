import os
import random
import paddle
from paddle.nn import Conv2D, MaxPool2D, Linear
import paddle.nn.functional as F
import numpy as np
from PIL import Image

import time
import gzip
import json

paddle.seed(0)
random.seed(0)
np.random.seed(0)

# 数据文件
datafile = './mnist.json.gz'
print('loading mnist dataset from {} ......'.format(datafile))
data = json.load(gzip.open(datafile))
train_set, val_set, eval_set = data

# 数据集相关参数，图片高度IMG_ROWS, 图片宽度IMG_COLS
IMG_ROWS = 28
IMG_COLS = 28
imgs, labels = train_set[0], train_set[1]
print("训练数据集数量: ", len(imgs))
assert len(imgs) == len(labels), \
    "length of train_imgs({}) should be the same as train_labels({})".format(
        len(imgs), len(labels))

from paddle.io import Dataset


class MnistDataset(Dataset):
    def __init__(self):
        self.IMG_COLS = 28
        self.IMG_ROWS = 28

    def __getitem__(self, idx):
        image = train_set[0][idx]
        image = np.array(image)
        image = image.reshape((1, IMG_ROWS, IMG_COLS)).astype('float32')
        label = train_set[1][idx]
        label = np.array(label)
        label = label.astype('int64')
        return image, label

    def __len__(self):
        return len(imgs)


# 调用加载数据的函数
dataset = MnistDataset()
train_loader = paddle.io.DataLoader(dataset, batch_size=100, shuffle=False, return_list=True)


# 定义模型结构
class MNIST(paddle.nn.Layer):
    def __init__(self):
        super(MNIST, self).__init__()

        # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
        self.conv1 = Conv2D(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2)
        # 定义池化层，池化层卷积核kernel_size为2，池化步长为2
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
        # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
        self.conv2 = Conv2D(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2)
        # 定义池化层，池化层卷积核kernel_size为2，池化步长为2
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)
        # 定义一层全连接层，输出维度是10
        self.fc = Linear(in_features=980, out_features=10)

    # 定义网络前向计算过程，卷积后紧接着使用池化层，最后使用全连接层计算最终输出
    def forward(self, inputs, label):
        x = self.conv1(inputs)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc(x)
        x = F.softmax(x)
        if label is not None:
            acc = paddle.metric.accuracy(input=x, label=label)
            return x, acc
        else:
            return x


# 在使用GPU机器时，可以将use_gpu变量设置成True
use_gpu = False
paddle.set_device('gpu:1') if use_gpu else paddle.set_device('cpu')

EPOCH_NUM = 5
BATCH_SIZE = 100

paddle.seed(0)


def train(model):
    model.train()

    BATCH_SIZE = 100
    # 定义学习率，并加载优化器参数到模型中

    total_steps = (int(50000 // BATCH_SIZE) + 1) * EPOCH_NUM
    lr = paddle.optimizer.lr.PolynomialDecay(learning_rate=0.01, decay_steps=total_steps, end_lr=0.001)
    # 使用Adam优化器
    opt = paddle.optimizer.Adam(learning_rate=lr, parameters=model.parameters())

    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            # 准备数据，变得更加简洁
            image_data = data[0].reshape([BATCH_SIZE, 1, 28, 28])
            label_data = data[1].reshape([BATCH_SIZE, 1])
            image = paddle.to_tensor(image_data)
            label = paddle.to_tensor(label_data)
            # if batch_id<10:
            # print(label.reshape([-1])[:10])
            # 前向计算的过程
            predict, acc = model(image, label)
            avg_acc = paddle.mean(acc)
            # 计算损失，使用交叉熵损失函数，取一个批次样本损失的平均值
            loss = F.cross_entropy(predict, label)
            avg_loss = paddle.mean(loss)

            # 每训练了200批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}, acc is {}".format(epoch_id, batch_id, avg_loss.numpy(),
                                                                            avg_acc.numpy()))

            # 后向传播，更新参数的过程
            avg_loss.backward()
            #更新参数
            opt.step()
            opt.clear_grad()

            # 保存模型参数和优化器的参数
            paddle.save(model.state_dict(), './checkpoint/mnist_epoch{}'.format(epoch_id) + '.pdparams')
            paddle.save(opt.state_dict(), './checkpoint/mnist_epoch{}'.format(epoch_id) + '.pdopt')
    print(opt.state_dict().keys())


time_start = time.time()
model = MNIST()
train(model)
time_end = time.time()

time_c= time_end - time_start   #运行所花时间

print('耗费总时间为', 'time cost', time_c, 's')
print(model.state_dict().keys())

params_path = "./checkpoint/mnist_epoch0"
# 在使用GPU机器时，可以将use_gpu变量设置成True
use_gpu = True
paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')


def train_again(model):
    model.train()

    # 读取参数文件
    params_dict = paddle.load(params_path + '.pdparams')
    opt_dict = paddle.load(params_path + '.pdopt')
    # 加载参数到模型
    model.set_state_dict(params_dict)

    EPOCH_NUM = 5
    BATCH_SIZE = 100
    # 定义学习率，并加载优化器参数到模型中
    total_steps = (int(50000 // BATCH_SIZE) + 1) * EPOCH_NUM
    lr = paddle.optimizer.lr.PolynomialDecay(learning_rate=0.01, decay_steps=total_steps, end_lr=0.001)
    # 使用Adam优化器
    opt = paddle.optimizer.Adam(learning_rate=lr, parameters=model.parameters())
    # 加载参数到优化器
    opt.set_state_dict(opt_dict)

    for epoch_id in range(1, EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            # 准备数据，变得更加简洁
            image_data = data[0].reshape([BATCH_SIZE, 1, 28, 28])
            label_data = data[1].reshape([BATCH_SIZE, 1])
            image = paddle.to_tensor(image_data)
            label = paddle.to_tensor(label_data)

            # 前向计算的过程
            predict, acc = model(image, label)

            avg_acc = paddle.mean(acc)
            # 计算损失，使用交叉熵损失函数，取一个批次样本损失的平均值
            loss = F.cross_entropy(predict, label)
            avg_loss = paddle.mean(loss)

            # 每训练了200批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}, acc is {}".format(epoch_id, batch_id, avg_loss.numpy(),
                                                                            avg_acc.numpy()))

            # 后向传播，更新参数的过程
            # print(opt.state_dict())
            avg_loss.backward()
            opt.step()
            opt.clear_grad()


model = MNIST()
train_again(model)