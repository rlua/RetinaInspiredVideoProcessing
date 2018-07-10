import torch
import numpy as np
import scipy.signal

#The following two functions and center_surround from https://github.com/smittal6/i3d/blob/master/src/utils.py

def gkern(kernlen=7, std=3, max =1):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = scipy.signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d * max


def kernels(size=5):
    m = gkern(kernlen=size, std = 5, max=0.5)
    #m1 = gkern(kernlen=size, std = 1, max=3.35) #sum of the kernel:  0.06041046812276252 for 7x7
    #m1 = gkern(kernlen=size, std = 1, max=1.88) #sum of the kernel:  0.04573110456660656 for 5x5
    m1 = gkern(kernlen=size, std = 1, max=2)
    #m1 = gkern(kernlen=size, std = 3, max=2)
    x = (m1 - m)
    x = x.reshape(1,size,size)

    #x=x/np.max(x)

    print("The filter: ")
    print(x)
    print("The sum of the kernel: ",np.sum(x))

    return x

_size=5
_pad=int((_size-1)/2)
_in_channels=3
_out_channels=3
#center_surround = torch.nn.Conv3d(1,1, kernel_size=(1,_size,_size), stride=1, padding=(0,_pad,_pad), bias=False)
#For RGB input
#center_surround = torch.nn.Conv3d(_in_channels,_out_channels, kernel_size=(1,_size,_size), stride=1, padding=(0,_pad,_pad), bias=False)
center_surround = torch.nn.Conv3d(_in_channels,_out_channels, kernel_size=(1,_size,_size), stride=1, padding=(0,_pad,_pad), bias=False,groups=_in_channels)

#center_surround.weight = torch.nn.parameter.Parameter(torch.from_numpy(kernels(size=_size)).view(1,_size,_size).repeat(1,1,1,1,1).float(),requires_grad=False)

#For RGB input, explicit
#kernel=np.concatenate((kernels(size=_size),np.zeros((1,_size,_size)),np.zeros((1,_size,_size)),\
#                       np.zeros((1,_size,_size)),kernels(size=_size),np.zeros((1,_size,_size)),\
#                       np.zeros((1,_size,_size)),np.zeros((1,_size,_size)),kernels(size=_size)),axis=0)
#center_surround.weight = torch.nn.parameter.Parameter(torch.from_numpy(kernel).view(_in_channels,_out_channels,1,_size,_size).float(),requires_grad=False)

#For RGB input, using groups argument in Conv3d
center_surround.weight = torch.nn.parameter.Parameter(torch.from_numpy(kernels(size=_size)).view(1,_size,_size).repeat(_out_channels,1,1,1,1).float(),requires_grad=False) #In conjunction with groups=3

print('center_surround.weight ',center_surround.weight)
