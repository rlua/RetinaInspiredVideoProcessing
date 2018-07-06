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
    m1 = gkern(kernlen=size, std = 1, max=1.88)
    #m1 = gkern(kernlen=size, std = 1, max=2)
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
center_surround = torch.nn.Conv3d(1,1, kernel_size=(1,_size,_size), stride=1, padding=(0,_pad,_pad), bias=False)

center_surround.weight = torch.nn.parameter.Parameter(torch.from_numpy(kernels(size=_size)).view(1,_size,_size).repeat(1,1,1,1,1).float(),requires_grad=False)

