import torch
import torch.nn as nn
import numpy as np
import itertools
# reference: https://github.com/mlomnitz/DiffJPEG


def diff_round(x):
    return torch.round(x) + (x - torch.round(x))**3


class block_splitting(nn.Module):
    def __init__(self):
        super(block_splitting, self).__init__()
        self.k = 8 # blocksize

    def forward(self, image): # image:(B,H,W)
        height, width = image.shape[1:3]
        batch_size = image.shape[0]
        image_reshaped = image.view(batch_size, height // self.k, self.k, -1, self.k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.contiguous().view(batch_size, -1, self.k, self.k) # (B,blocknum,blocksize,blocksize)
    

class dct_8x8(nn.Module):
    def __init__(self):
        super(dct_8x8, self).__init__()
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos((2 * y + 1) * v * np.pi / 16)
        alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
        self.tensor =  nn.Parameter(torch.from_numpy(tensor).float())
        self.scale = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha) * 0.25).float() )
        
    def forward(self, image):
        image = image - 127.5
        result = self.scale * torch.tensordot(image, self.tensor, dims=2)
        result.view(image.shape)
        return result


class quantize(nn.Module):
    def __init__(self, rounding, factor, y_table, c_table):
        super(quantize, self).__init__()
        self.rounding = rounding
        self.factor = factor
        self.y_table = nn.Parameter(torch.Tensor(y_table), requires_grad=False)
        self.c_table = nn.Parameter(torch.Tensor(c_table), requires_grad=False)

    def forward(self, image, channel='y'):
        table = self.y_table if channel=='y' else self.c_table
        image = image.float() / (table * self.factor)
        image = self.rounding(image)
        return image


class dequantize(nn.Module):
    def __init__(self, factor, y_table, c_table):
        super(dequantize, self).__init__()
        self.factor = factor
        self.y_table = nn.Parameter(torch.Tensor(y_table), requires_grad=False)
        self.c_table = nn.Parameter(torch.Tensor(c_table), requires_grad=False)

    def forward(self, image, channel='y'):
        table = self.y_table if channel=='y' else self.c_table
        return image * (table * self.factor)


class idct_8x8(nn.Module):
    def __init__(self):
        super(idct_8x8, self).__init__()
        alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
        self.alpha = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha)).float())
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos((2 * v + 1) * y * np.pi / 16)
        self.tensor = nn.Parameter(torch.from_numpy(tensor).float())

    def forward(self, image):
        image = image * self.alpha
        result = 0.25 * torch.tensordot(image, self.tensor, dims=2) + 127.5
        result.view(image.shape)
        return result


class block_merging(nn.Module):
    def __init__(self):
        super(block_merging, self).__init__()
        self.k = 8 # blocksize
        
    def forward(self, patches, height, width):
        batch_size = patches.shape[0]
        image_reshaped = patches.view(batch_size, height//self.k, width//self.k, self.k, self.k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.contiguous().view(batch_size, height, width)


class DCTQuantify(nn.Module):
    def __init__(self, height, width, factor=1, color_space='rgb'): # QF=50
        super().__init__()
        y_table = np.array(
            [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], dtype=np.float32).T

        c_table = np.ones((8, 8), dtype=np.float32) * 99
        c_table[:4,:4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66],
                                   [24, 26, 56, 99], [47, 66, 99, 99]], dtype=np.float32).T

        self.height, self.width, self.color_space = height, width, color_space
        self.block_split = block_splitting()
        self.dct_8x8 = dct_8x8()
        self.quantize = quantize(diff_round, factor, y_table, c_table)
        self.dequantize = dequantize(factor, y_table, c_table)
        self.idct = idct_8x8()
        self.merging = block_merging()
    
    def set_factor(self, factor):
        self.quantize.factor = factor
        self.dequantize.factor = factor
    
    def forward(self, input, factor=None): # input:-1~1,(B,C,H,W)
        if factor: self.set_factor(factor)
        else: return input
        input = (input+1)*127.5
        channellist = ['y','u','v']
        s = []
        for i in range(input.shape[1]):
            if i>0: self.set_factor(factor/2)
            comp = self.block_split(input[:,i])
            comp = self.dct_8x8(comp)
            comp = self.quantize(comp, channel=channellist[i])
            comp = self.dequantize(comp, channel=channellist[i])
            comp = self.idct(comp)
            sttr = self.merging(comp, self.height, self.width)
            s.append(sttr)
        image = torch.stack(s, dim=1)
        image = image.clamp(0,255)/127.5-1
        return image
