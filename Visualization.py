from sklearn.metrics import confusion_matrix
from get_dataset import test_dataset_prepare
from matplotlib import pyplot as plt
import numpy as np
from my_model import Res_Base_complex_model,Base_complex_model
import torch
from get_dataset import train_dataset_prepare,test_dataset_prepare
from torch.utils.data import TensorDataset,DataLoader
from matplotlib import rcParams
from torch import nn

def draw_confusion_matrix(model:nn.Module,device,classes:int=10,is_save:bool=False):
    model.eval()
    X, Y = test_dataset_prepare(classes)
    Y_pred,Y_true = [],[]
    test_dataset = TensorDataset(torch.tensor(X), torch.tensor(Y)) #打包测试集
    test_data_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=True)
    with torch.no_grad():
        for X_,Y_ in test_data_loader:
            if torch.cuda.is_available():
                X_ = X_.to(device)
                Y_ = Y_.to(device)

            out = model(X_)
            Y_out = out.argmax(dim = 1)
            Y_true = Y_true+Y_.cpu().numpy().tolist()
            Y_pred=Y_pred+Y_out.cpu().numpy().tolist()
    my_confusion_matrix = confusion_matrix(Y_true, Y_pred)
    # confusion_matrix
    classes = range(classes)
    my_confusion_matrix = np.array(my_confusion_matrix)  # 输入特征矩阵
    proportion = []
    length = len(my_confusion_matrix)
    # print(length)
    for i in my_confusion_matrix: # 计算百分比
        for j in i:
            temp = j / (np.sum(i))
            proportion.append(temp)
    # print(np.sum(confusion_matrix[0]))
    # print(proportion)
    pshow = []
    for i in proportion:
        pt = "%.1f%%" % (i * 100)
        pshow.append(pt)
    proportion = np.array(proportion).reshape(length, length)  # reshape(列的长度，行的长度)
    pshow = np.array(pshow).reshape(length, length)
    # print(pshow)
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
    }
    plt.figure(dpi=200)
    rcParams.update(config)
    plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵
    # (改变颜色：'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd',
    # 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn')
    # plt.title('confusion_matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    thresh = my_confusion_matrix.max() / 2.
    # iters = [[i,j] for i in range(len(classes)) for j in range((classes))]

    iters = np.reshape([[[i, j] for j in range(length)] for i in range(length)], (my_confusion_matrix.size, 2))
    for i, j in iters:
        if (i == j):
            plt.text(j, i - 0.12, format(my_confusion_matrix[i, j]), va='center', ha='center', fontsize=10, color='white',
                     weight=5)  # 显示对应的数字
            plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=10, color='white')
        else:
            plt.text(j, i - 0.12, format(my_confusion_matrix[i, j]), va='center', ha='center', fontsize=10)  # 显示对应的数字
            plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=10)

    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predict label', fontsize=16)
    plt.tight_layout()
    if is_save:
        plt.savefig('confusion_matrix.png')
    plt.show()



if __name__ =='__main__':
    device = torch.device('cuda:0')
    batch_size =32
    model = Res_Base_complex_model()
    model.load_state_dict(torch.load('model_weight/Res_complex_conv.pt'))
    model = model.to(device)
    draw_confusion_matrix(model,device,10,is_save=True)




