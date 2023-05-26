import torch
import torch.nn as nn
import numpy as np


class ComplexConv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride = 1,padding=1,dilation=1,groups=1,bias=True):
        super(ComplexConv, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.padding = padding
        self.conv_re = nn.Conv2d(in_channels,out_channels,(1,kernel_size),stride,(0,padding),dilation,groups,bias)
        self.conv_im = nn.Conv2d(in_channels,out_channels,(1,kernel_size),stride,(0,padding),dilation,groups,bias)

        # self.conv_re = nn.Conv1d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias)
        # self.conv_im = nn.Conv1d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias)

    def forward(self,x):
        # x_real = x[:,0:x.shape[1]//2,:]
        # x_img = x[:,x.shape[1]//2:x.shape[1],:]
        # real = self.conv_re(x_real)-self.conv_im(x_img)
        # imaginary = self.conv_re(x_img)+self.conv_im(x_real)
        # output = torch.cat((real,imaginary),dim=1)
        x_real = x[:,:,0:x.shape[2]//2,:]
        x_img = x[:,:,x.shape[2]//2:x.shape[2],:]
        real = self.conv_re(x_real)-self.conv_im(x_img)
        imaginary = self.conv_re(x_img)+self.conv_im(x_real)
        output = torch.cat((real,imaginary),dim=2)
        return output

