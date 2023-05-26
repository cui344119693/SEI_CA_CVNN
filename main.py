# This is a sample Python script.
import os

import numpy as np
import torch
import random
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from get_dataset import train_dataset_prepare, test_dataset_prepare
from my_model import Base_complex_model, Res_Base_complex_model, CBAM_Res_Base_complex_model,CBAM_Residual_Base_complex_model
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
from Visualization import draw_confusion_matrix
from colorama import Fore, Back, Style

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Config:
    """
    参数配置
    """

    def __init__(
            self,
            batch_size: int = 32,
            test_batch_size: int = 32,
            epochs: int = 40,
            lr: float = 0.001,
            n_classes: int = 10,
            save_path: str = 'model_weight/test.pt',
            device: int = 0,
            rand_num: int = 50):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.n_classes = n_classes
        self.save_path = save_path
        self.device = device
        self.rand_num = rand_num


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.cuda.manual_seed_all(seed)  # all gpu
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现
    torch.backends.cudnn.benchmark = False


def train(model: nn.Module, loss, optimizer, data_loader, epoch, device):
    model.train()
    correct = 0
    classifier_loss = 0
    for X, label in data_loader:
        label = label.long()
        if torch.cuda.is_available():
            X = X.to(device)
            label = label.to(device)
        optimizer.zero_grad()
        pred = model(X)
        batch_loss = loss(pred, label)
        batch_loss.backward()
        optimizer.step()
        classifier_loss += batch_loss
        pred_label = pred.argmax(dim=1, keepdim=True)
        correct += pred_label.eq(label.view_as(pred_label)
                                 ).sum().item()  # 计算一批数据中预测正确的个数
    classifier_loss /= len(data_loader)
    print('Train epoch:{}\t classifier_loss:{:.6f},accuracy:{}/{}({:.1f}%)\n'.format(epoch,
          classifier_loss, correct, len(data_loader.dataset), 100 * correct / len(data_loader.dataset)))
    return classifier_loss


def test(model: nn.Module, loss, data_loader, epoch, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X, label in data_loader:
            label = label.long()
            if torch.cuda.is_available():
                X = X.to(device)
                label = label.to(device)
            pred = model(X)
            batch_loss = loss(pred, label)
            test_loss += batch_loss
            pred_label = pred.argmax(dim=1, keepdim=True)
            correct += pred_label.eq(label.view_as(pred_label)
                                     ).sum().item()  # 计算一批数据中预测正确的个数
    test_loss /= len(data_loader)
    print('Validation epoch:{}\t test_loss:{:.6f},accuracy:{}/{}({:.1f}%)\n'.format(epoch,
          test_loss, correct, len(data_loader.dataset), 100 * correct / len(data_loader.dataset)))
    return test_loss


def main():
    """
    主函数
    """
    conf = Config()
    device = torch.device('cuda:' + str(conf.device))
    RANDOM_SEED = 300
    set_seed(RANDOM_SEED)
    test_x = np.arange(conf.epochs)
    test_y = np.zeros(conf.epochs)

    x_train, x_val, y_train, y_val = train_dataset_prepare(
        conf.n_classes, conf.rand_num)
    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    train_data_loader = DataLoader(
        train_dataset, conf.batch_size, shuffle=True)
    val_dataset = TensorDataset(torch.tensor(x_val), torch.tensor(y_val))
    val_data_loader = DataLoader(
        val_dataset, conf.batch_size, shuffle=True)

    model = CBAM_Residual_Base_complex_model()
    # model = CBAM_Res_Base_complex_model()

    if torch.cuda.is_available():
        model = model.to(device)
    print(model)
    loss = nn.NLLLoss()
    if torch.cuda.is_available():
        loss = loss.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=conf.lr, weight_decay=0)
    # 开始训练
    current_min_test_loss = 1
    for epoch in range(conf.epochs):
        train_loss = train(
            model,
            loss,
            optim,
            train_data_loader,
            epoch,
            device)
        # if epoch % 5 == 0:
        test_loss = test(model, loss, val_data_loader, epoch, device)
        test_y[epoch] = train_loss
        if test_loss < current_min_test_loss:
            print(Fore.RED + Back.CYAN +
                  '测试集损失从{}减小到了{}，保存新的模型参数'.format(
                      current_min_test_loss,
                      test_loss) + Style.RESET_ALL)
            current_min_test_loss = test_loss
            torch.save(model.state_dict(), conf.save_path)
        else:
            print('测试集性能未提升')
        print('-------------------------------------------------------------------')

    plt.figure(1)
    plt.plot(test_x, test_y)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(conf.save_path.replace('model_weight/', '').replace('.pt', ''))
    plt.savefig(conf.save_path.replace('.pt', '.png'))

    plt.show()

    draw_confusion_matrix(model, device, 10)


def moudle_test():
    device = torch.device('cuda:0')
    model = Res_Base_complex_model()
    model.load_state_dict(torch.load('model_weight/test.pt'))
    model = model.to(device)
    model.eval()  # 固定dropout和批次归一化
    print(model)
    # 支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    X_train_label, X_val, Y_train_label, Y_val = train_dataset_prepare(10, 300)

    x_axis = np.arange(4800)
    # print(x_axis.shape)
    # print(((x[0][:]).T)[0].shape)
    print(X_train_label.shape)
    print(Y_train_label.shape)
    print(Y_val.shape)

    i = random.randint(0, X_train_label.shape[0])  # 选择第i组数据
    # print(i)
    data = X_train_label[i]
    data = torch.tensor(data)
    data = data.unsqueeze(0)
    data = data.to(device)
    label = Y_train_label[i]
    pred = model(data)
    pred_lable = pred.argmax(dim=1, keepdim=True)
    plt.figure(1)
    plt.xlabel('x')
    plt.ylabel('I/Q')
    plt.title('第' + str(i) + '组数据  ' + 'class:' + str(Y_train_label[i]))
    plt.plot(x_axis, ((X_train_label[i][0][:]))[0], color='r', label='I')
    plt.plot(x_axis, ((X_train_label[i][0][:]))[1], color='g', label='Q')
    plt.text(100, 0.5, 'Pred_class:' +
             str(pred_lable.item()), fontsize=22, color="b")
    plt.legend()
    # ----------------------------------------------------
    plt.figure(2)
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    plt.sca(ax1)
    plt.xlabel('x')
    plt.ylabel('I')
    plt.title('第' + str(i) + '组数据  ' + 'class:' + str(Y_train_label[i]))
    plt.plot(x_axis, ((X_train_label[i][0][:]))[0], color='r')
    plt.text(100, 0.5, 'Pred_class:' +
             str(pred_lable.item()), fontsize=22, color="b")
    plt.sca(ax2)
    plt.xlabel('x')
    plt.ylabel('Q')
    plt.plot(x_axis, ((X_train_label[i][0][:]))[1], color='g')
    plt.text(100, 0.5, 'Pred_class:' +
             str(pred_lable.item()), fontsize=22, color="b")
    plt.show()


if __name__ == '__main__':
    main()
    # moudle_test()
