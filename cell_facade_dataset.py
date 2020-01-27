import os

import numpy as np

from PIL import Image

import six

from io import BytesIO

import pickle

import json

import skimage.io as io

from chainer.dataset import dataset_mixin

import matplotlib.pyplot as plt



# download `BASE` dataset from http://cmp.felk.cvut.cz/~tylecr1/facade/

class FacadeDataset(dataset_mixin.DatasetMixin):

#    def __init__(self, dataDir='./facade/cell_image4', data_range=(1,300)):

    def __init__(self, dataDir, data_range): 

# 180927 by kishi : main()で値を指定している。ここの値が優先されるので、変更前のコードだとmain()で値を変えても変わらないよ。値管理は呼び出す側で行うこと！　

        print("load dataset start")

        print("    from: %s" % dataDir)

        print("    range: [%d, %d)" % (data_range[0], data_range[1]))

        self.dataDir = dataDir

        self.dataset = []

        for i in range(data_range[0], data_range[1]):

            img = Image.open(dataDir + "/c%04d.jpg" % i)

            label = Image.open(dataDir + "/c%04d.png" % i)

            w, h = img.size

            r = 286 / float(min(w, h))

            # resize images so that min(w, h) == 286

            img = img.resize((int(r * w), int(r * h)), Image.BILINEAR)

#            label = label.resize((int(r * w), int(r * h)), Image.NEAREST)

            label = label.resize((int(r * w), int(r * h)), Image.BILINEAR) 
# 180927 by kishi : resize時の精度向上

            img = np.asarray(img).astype("f").transpose(2, 0, 1) / 128.0 - 1.0

#            label = np.asarray(label) - 1  # [0, 12)


 #           label = np.zeros((12, img.shape[1], img.shape[2])).astype("i")

            label = np.asarray(label).astype("f").transpose(2, 0, 1) / 128.0 - 1.0
# 180927 by kishi : 型タイプをfloatに変更！

 #           for j in range(12):
 #               label[j, :] = label_ == j

            self.dataset.append((img, label))

        print("load dataset done")

    def __len__(self):

        return len(self.dataset)

    # return (label, img)

    def get_example(self, i, crop_width=256):

        _, h, w = self.dataset[i][0].shape

        x_l = np.random.randint(0, w - crop_width)

        x_r = x_l + crop_width

        y_l = np.random.randint(0, h - crop_width)

        y_r = y_l + crop_width

#        return self.dataset[i][1][:, y_l:y_r, x_l:x_r], self.dataset[i][0][:, y_l:y_r, x_l:x_r]

        return self.dataset[i][0][:, y_l:y_r, x_l:x_r], self.dataset[i][1][:, y_l:y_r, x_l:x_r] # change by KK @180927

# 180927 by kishi : やはりimgとlabelを取り違えていそうなので、返す順序を変更



