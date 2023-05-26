import torch
import numpy as np
import random
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def train_dataset_prepare(num, random_num):
    x = np.load(f'./FS-SEI_4800/Dataset/X_train_{num}Class.npy')
    y = np.load(f'./FS-SEI_4800/Dataset/Y_train_{num}Class.npy')
    x = x.astype(np.float32)
    y = y.astype(np.uint8)
    x=x.transpose(0,2,1)
    x = x[:,np.newaxis,:,:]  # 增加一个通道维度
    # 数据归一化处理
    min_value = x.min()
    max_value = x.max()
    x = (x-min_value)/(max_value-min_value)
    # 分割数据集和验证集
    x_train_label, x_val, y_train_label, y_val = train_test_split(
        x, y, test_size=0.15, random_state=random_num)

    return x_train_label, x_val, y_train_label, y_val


def test_dataset_prepare(num):
    x = np.load(f'./FS-SEI_4800/Dataset/X_test_{num}Class.npy')
    y = np.load(f'./FS-SEI_4800/Dataset/Y_test_{num}Class.npy')
    x = x.transpose(0,2,1)
    x = x[:,np.newaxis,:,:]  # 增加一个通道维度
    min_value = x.min()
    max_value = x.max()
    x = (x-min_value)/(max_value-min_value)
    x = x.astype(np.float32)
    y = y.astype(np.uint8)
    return x,y


if __name__ == '__main__':
    # 支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    X_train_label,  X_val, Y_train_label,  Y_val = train_dataset_prepare(10, 300)

    x_axis = np.arange(4800)
    # print(x_axis.shape)
    # print(((x[0][:]).T)[0].shape)
    print(X_train_label.shape)
    print(Y_train_label.shape)
    print(Y_val.shape)

    i = random.randint(0, X_train_label.shape[0])  # 选择第i组数据
    # print(i)
    # i=1000 #选择第i组数据
    plt.figure(1)
    plt.xlabel('x')
    plt.ylabel('I/Q')
    plt.title('第' + str(i) + '组数据  '+'class:'+str(Y_train_label[i]))
    plt.plot(x_axis, ((X_train_label[i][0][:]))[0], color='r', label='I')
    plt.plot(x_axis, ((X_train_label[i][0][:]))[1], color='g', label='Q')
    # plt.text(100, 0.5, 'class:' + str(Y_train_label[i]), fontsize=22, color="b")
    plt.legend()
    # ----------------------------------------------------
    plt.figure(2)
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    plt.sca(ax1)
    plt.xlabel('x')
    plt.ylabel('I')
    plt.title('第' + str(i) + '组数据  '+'class:'+str(Y_train_label[i]))
    plt.plot(x_axis, ((X_train_label[i][0][:]))[0], color='r')
    # plt.text(100, 0.5, 'class:' + str(Y_train_label[i]), fontsize=22, color="b")
    plt.sca(ax2)
    plt.xlabel('x')
    plt.ylabel('Q')
    plt.plot(x_axis, ((X_train_label[i][0][:]))[1], color='g')
    # plt.text(100, 0.5, 'class:' + str(Y_train_label[i]), fontsize=22, color="b")
    plt.show()
