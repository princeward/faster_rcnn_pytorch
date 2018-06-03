import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from utils.blob import im_list_to_blob
from network import Conv2d
import network


class Disparity16(nn.Module):
    """Adopted from VGG16
    """
    def __init__(self, bn=False):
        super(Disparity16, self).__init__()

        self.conv1 = nn.Sequential(Conv2d(1, 16, 3, same_padding=True, bn=bn),
                                   Conv2d(16, 16, 3, same_padding=True, bn=bn),
                                   nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(Conv2d(16, 32, 3, same_padding=True, bn=bn),
                                   Conv2d(32, 32, 3, same_padding=True, bn=bn),
                                   nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(Conv2d(32, 64, 3, same_padding=True, bn=bn),
                                   Conv2d(64, 64, 3, same_padding=True, bn=bn),
                                   Conv2d(64, 64, 3, same_padding=True, bn=bn),
                                   nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(Conv2d(64, 128, 3, same_padding=True, bn=bn),
                                   Conv2d(128, 128, 3, same_padding=True, bn=bn),
                                   Conv2d(128, 128, 3, same_padding=True, bn=bn),
                                   nn.MaxPool2d(2))
        self.conv5 = nn.Sequential(Conv2d(128, 128, 3, same_padding=True, bn=bn),
                                   Conv2d(128, 128, 3, same_padding=True, bn=bn),
                                   Conv2d(128, 128, 3, same_padding=True, bn=bn))
        network.set_trainable(self.conv1, requires_grad=True)
        network.set_trainable(self.conv2, requires_grad=True)
        network.set_trainable(self.conv3, requires_grad=True)
        network.set_trainable(self.conv4, requires_grad=True)
        network.set_trainable(self.conv5, requires_grad=True)

    def forward(self, im_data):
        # im_data, im_scales = get_blobs(image)
        # im_info = np.array(
        #     [[im_data.shape[1], im_data.shape[2], im_scales[0]]],
        #     dtype=np.float32)
        # data = Variable(torch.from_numpy(im_data)).cuda()
        # x = data.permute(0, 3, 1, 2)

        x = self.conv1(im_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def load_from_npz(self, params):
        # params = np.load(npz_file)
        own_dict = self.state_dict()
        for name, val in own_dict.items():
            i, j = int(name[4]), int(name[6]) + 1
            ptype = 'weights' if name[-1] == 't' else 'biases'
            key = 'conv{}_{}/{}:0'.format(i, j, ptype)
            param = torch.from_numpy(params[key])
            if ptype == 'weights':
                param = param.permute(3, 2, 0, 1)
            val.copy_(param)

    # def load_from_npy_file(self, fname):
    #     own_dict = self.state_dict()
    #     params = np.load(fname).item()
    #     for name, val in own_dict.items():
    #         # # print name
    #         # # print val.size()
    #         # # print param.size()
    #         # if name.find('bn.') >= 0:
    #         #     continue
    #
    #         i, j = int(name[4]), int(name[6]) + 1
    #         ptype = 'weights' if name[-1] == 't' else 'biases'
    #         key = 'conv{}_{}'.format(i, j)
    #         param = torch.from_numpy(params[key][ptype])
    #
    #         if ptype == 'weights':
    #             param = param.permute(3, 2, 0, 1)
    #
    #         val.copy_(param)


#if __name__ == '__main__':
#    vgg = VGG16()
#    vgg.load_from_npy_file('/media/longc/Data/models/VGG_imagenet.npy')