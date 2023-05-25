import torch
from torch import nn
import torch.nn.functional as F
from complexcnn import ComplexConv





class ChannelAttention1d(nn.Module):
    def __init__(self):
        pass

    def forward(self,x):
        pass


class SpatialAttention1d(nn.Module):
    def __init__(self):
        pass

    def forward(self,x):
        pass


class ChannelAttention(nn.Module):
    """
    paper: CBAM: Convolutional Block Attention Module
    :arg in_planes :输入通道数
    :return 通道维度上的加权系数
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # arg 1 is output size
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    paper: CBAM: Convolutional Block Attention Module
    :arg kernel_size 卷积核大小
    :return 2D维度上的加权系数

    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CA_Block(nn.Module):
    """
    Coordinate attention
    channel 为输入数据的通道数
    reduction 为模块中间层的通道衰减因子
    pic：pic/CoordinateAttention.png
    """
    def __init__(self,channel,reduction = 16):
        super(CA_Block, self).__init__()
        self.conv_1x1 = nn.Conv2d(in_channels=channel,out_channels=channel//reduction)
        self.bn = nn.BatchNorm2d(channel//reduction)
        self.F_H = nn.Conv2d(in_channels=channel//reduction,out_channels=channel)
        self.F_W = nn.Conv2d(in_channels=channel//reduction,out_channels=channel)
    def forward(self,x):
        _,_,h,w = x.size() # input b,c,h,w
        x_h = torch.mean(x,dim=3,keepdim=True).permute(0,1,3,2) # b,c,h,1 ->b,c,1,h
        x_w = torch.mean(x,dim=2,keepdim=True) # b,c,1,w
        x_cat_conv_relu = nn.ReLU(self.bn(self.conv_1x1(torch.cat((x_h,x_w),dim=3)))) # b,c,1,(h+w)
        x_cat_conv_split_h,x_cat_conv_split_w = x_cat_conv_relu.split([h,w],3) # b,c,1,h and b,c,1,w
        s_h = nn.Sigmoid(self.F_H(x_cat_conv_split_h.permute(0,1,3,2)))  # b,c,h,1
        s_w = nn.Sigmoid(self.F_W(x_cat_conv_split_w)) # b,c,1,w

        out = x*s_h.expand_as(x)*s_w.expand_as(x) # output b,c,h,w
        return out





class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                  conv_kernel_size=3,BN_num_features=128,MP_kernel_size=2):
        super().__init__()
        self.conv1 = Complex_block(
            conv_in_channels=input_channels,
            conv_out_channels=num_channels,
            conv_kernel_size=conv_kernel_size,
            BN_num_features=BN_num_features,
            MP_kernel_size=MP_kernel_size)

        self.conv2 = Complex_block(
            conv_in_channels=num_channels,
            conv_out_channels=num_channels,
            conv_kernel_size=conv_kernel_size,
            BN_num_features=BN_num_features,
            MP_kernel_size=MP_kernel_size)

        self.conv3 = ComplexConv(
            in_channels=input_channels,
            out_channels=num_channels,
            kernel_size=conv_kernel_size,
            stride=4)


    def forward(self, X):
        Y = self.conv1(X)
        Y = self.conv2(Y)
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return Y


class Complex_block(nn.Module):
    def __init__(
            self,
            conv_in_channels,
            conv_out_channels,
            conv_kernel_size,
            BN_num_features,
            MP_kernel_size,
            stride=1):
        super(Complex_block, self).__init__()
        self.conv = ComplexConv(
            in_channels=conv_in_channels,
            out_channels=conv_out_channels,
            kernel_size=conv_kernel_size,
            stride=stride)
        self.batchnorm = nn.BatchNorm1d(num_features=BN_num_features)
        self.maxpool = nn.MaxPool1d(kernel_size=MP_kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        x = self.maxpool(x)
        return x


class Base_complex_model(nn.Module):
    def __init__(self):
        super(Base_complex_model, self).__init__()
        self.complex_block1 = Complex_block(
            conv_in_channels=1,
            conv_out_channels=64,
            conv_kernel_size=3,
            BN_num_features=128,
            MP_kernel_size=2)
        self.complex_block2 = Complex_block(
            conv_in_channels=64,
            conv_out_channels=64,
            conv_kernel_size=3,
            BN_num_features=128,
            MP_kernel_size=2)
        self.complex_block3 = Complex_block(
            conv_in_channels=64,
            conv_out_channels=64,
            conv_kernel_size=3,
            BN_num_features=128,
            MP_kernel_size=2)
        self.complex_block4 = Complex_block(
            conv_in_channels=64,
            conv_out_channels=64,
            conv_kernel_size=3,
            BN_num_features=128,
            MP_kernel_size=2)
        self.complex_block5 = Complex_block(
            conv_in_channels=64,
            conv_out_channels=64,
            conv_kernel_size=3,
            BN_num_features=128,
            MP_kernel_size=2)
        self.complex_block6 = Complex_block(
            conv_in_channels=64,
            conv_out_channels=64,
            conv_kernel_size=3,
            BN_num_features=128,
            MP_kernel_size=2)
        self.complex_block7 = Complex_block(
            conv_in_channels=64,
            conv_out_channels=64,
            conv_kernel_size=3,
            BN_num_features=128,
            MP_kernel_size=2)
        self.complex_block8 = Complex_block(
            conv_in_channels=64,
            conv_out_channels=64,
            conv_kernel_size=3,
            BN_num_features=128,
            MP_kernel_size=2)
        self.complex_block9 = Complex_block(
            conv_in_channels=64,
            conv_out_channels=64,
            conv_kernel_size=3,
            BN_num_features=128,
            MP_kernel_size=2)

        self.flatten = nn.Flatten()
        self.linear1 = nn.LazyLinear(512)
        self.linear2 = nn.LazyLinear(128)
        self.linear3 = nn.LazyLinear(10)

    def forward(self, x):
        x = self.complex_block1(x)
        x = self.complex_block2(x)
        x = self.complex_block3(x)
        x = self.complex_block4(x)
        x = self.complex_block5(x)
        x = self.complex_block6(x)
        x = self.complex_block7(x)
        x = self.complex_block8(x)
        x = self.complex_block9(x)

        x = self.flatten(x)
        x = self.linear1(x)
        x = F.relu(x)

        x = self.linear2(x)
        x = F.relu(x)

        x = self.linear3(x)

        return F.log_softmax(x, dim=1)

class Res_Base_complex_model(nn.Module):
    def __init__(self):
        super(Res_Base_complex_model, self).__init__()
        self.complex_block1 = Complex_block(
            conv_in_channels=1,
            conv_out_channels=64,
            conv_kernel_size=3,
            BN_num_features=128,
            MP_kernel_size=2)
        self.res_block1=Residual(64,64)
        self.res_block2 = Residual(64, 64)
        self.res_block3 = Residual(64, 64,conv_kernel_size=2)
        self.res_block4 = Residual(64, 64,conv_kernel_size=2)


        self.flatten = nn.Flatten()
        self.linear1 = nn.LazyLinear(512)
        self.linear2 = nn.LazyLinear(128)
        self.linear3 = nn.LazyLinear(10)

    def forward(self, x):
        x = self.complex_block1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)


        x = self.flatten(x)
        x = self.linear1(x)
        x = F.relu(x)

        x = self.linear2(x)
        x = F.relu(x)

        x = self.linear3(x)

        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    model = SpatialAttention()
    print(model)
    test_input = torch.randn((32, 64, 640,640))
    out = model(test_input)
    print(out)
