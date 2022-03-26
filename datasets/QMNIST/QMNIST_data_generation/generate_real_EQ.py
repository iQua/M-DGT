#!/usr/bin/env python
# -*- coding: utf-8 -*-


''' python inherent libs '''
import sys
import os
import struct
import operator as op
import itertools

''' third parts libs '''
import numpy as np
from numpy.random import *
from scipy.misc import *
import operator as op
from dataset_utils import read
from PIL import Image as image
from PIL import ImageDraw

''' local custom libs '''


mnist_path = '/media/flyvideo/Maxtor1/工作区/research/mine_papers/attention_multiviews_searching/coding/DataSets/MNIST_data'
n_syn_imgs = 5000
#dest_path = '/media/flyvideo/Maxtor1/工作区/research/mine_papers/attention_multiviews_searching/coding/DataSets/MNIST_data/msbist'
dest_path = '/media/flyvideo/Maxtor1/工作区/research/mine_papers/attention_multiviews_searching/coding/DataSets/MNIST_data/mnist_rl'

itemset = sorted([x for x in read(dataset="training", path=mnist_path)] +
                 [x for x in read(dataset="testing", path=mnist_path)], key=op.itemgetter(0))


source_imgs_path = "/media/flyvideo/Maxtor1/工作区/DL_sources_factory/COCO/COCO-2017/test2017"
source_imgs_files = [os.path.join(source_imgs_path, img) for img in os.listdir(source_imgs_path)]

"""question types
     1) colors: no same target number
     2) objects (spatial relation):
     3) numbers:
     4) location:
"""

numPools = []
for i in range(9):
    count = 0
    for j in range(len(itemset)):
        if itemset[j][0] != i:
            break
        count += 1
    numPools.append(itemset[:count])
    itemset = itemset[count:]
numPools.append(itemset)

colorNames = ['red', 'green', 'blue', 'yellow', 'white']
colorMap = {'red': [0x92, 0x10, 0x10], 'green': [0x10, 0xA6, 0x51], 'blue': [
    0x10, 0x62, 0xC0], 'yellow': [0xFF, 0xF1, 0x4E], 'white': [0xE0, 0xE0, 0xE0]}
for k in list(colorMap.keys()):
    colorMap[k] = np.array(colorMap[k])


def generate_query_patch():
    """ For each number from 0-9, generate colored patch """
    w = h = 28
    for i in range(10):
        query_img_dir = dest_path + '/questions/%d' % i
        if not os.path.exists(query_img_dir):
            os.makedirs(query_img_dir)
        sampled_num_mnist = numPools[i][0][1]
        for name in colorNames:
            newImg = np.zeros((w, h, 4), dtype=np.uint8)
            color = colorMap[name]
            for i in range(w):
                for j in range(h):
                    if sampled_num_mnist[i, j] != 0:
                        newImg[i, j, 0:3] = color * \
                            (sampled_num_mnist[i, j]) / 255.
                        newImg[i, j, 3] = sampled_num_mnist[i, j]

            newImg[newImg > 255] = 255
            newImg[newImg < 0] = 0
            newImg = newImg.astype(np.uint8)

            img = image.fromarray(newImg)
            # img = image.fromarray(background)
            # img.paste(img_, None, img_)
            img = img.convert('RGB')
            img.save(os.path.join(query_img_dir, "%s.png" % name))


def generateSimpleColorQuestionInfo(minnum=4, maxnum=6):
    #choose_id = np.random.rand()
    # if choose_id < 0.35:
    #     numbers = [i for i in range(6)] * 2
    #     shuffle(numbers)

    # elif choose_id < 0.65:
    #     numbers = [i for i in range(4, 10)] * 2
    #     shuffle(numbers)

    # elif choose_id < 0.9:
    #     numbers = [i for i in list(range(0, 4)) + list(range(6, 10))] * 2
    #     shuffle(numbers)

    # else:
    #     numbers = [[i, i] for i in range(10)]
    #     shuffle(numbers)
    #     numbers = list(itertools.chain.from_iterable(numbers))
    numbers = [[i, i, i, i] for i in range(10)]
    shuffle(numbers)
    numbers = list(itertools.chain.from_iterable(numbers))

    maxnum = np.argmax(multinomial(1, [1. / (maxnum - minnum + 1)]
                                   * (maxnum - minnum + 1)), axis=0) + minnum
    colors = np.argmax(multinomial(1, [1. / len(colorNames)] * len(colorNames), maxnum), axis=1)

    res = []
    posset = set()

    # change the condition that the same numbers are in same colors
    length_nums = len(numbers[:maxnum])
    for i in range(length_nums - 1):
        for j in range(i + 1, length_nums):
            if numbers[i] == numbers[j]:
                if colors[i] == colors[j]:
                    colors[j] = (colors[j] + 1) % len(colorNames)

    for i in range(maxnum):
        res.append((numbers[i], colorNames[colors[i]], colorMap[colorNames[colors[i]]]))

    return res

scales = []


def renderImage(imgPath, imgIdx, questionInfoList, colorParam=(0, 10), scaleParam=(0.7, 2.5), num_dist=17):
    # 打开一个jpg图像文件，注意路径要改成你自己的:
    im = image.open(source_imgs_files[imgIdx])
    # 获得图像尺寸:
    w, h = im.size
    sampleSize = (28, 28)
    newImg = np.zeros((h, w, 4), dtype=np.uint8)
    # background = np.zeros((w, h, 4), dtype=np.uint8)
    background = im
    generated_bboxs = list()

    # newImg[:]=255

    # # put distractors
    # for n in range(num_dist):
    #     sampleNum = int(uniform() * 10)
    #     sampleIdx = int(uniform() * len(numPools[sampleNum]))

    #     color = colorMap[colorNames[int(uniform() * 5)]]

    #     colorNoise = normal(colorParam[0], colorParam[1])
    #     color = color + colorNoise
    #     color[color < 0] = 0
    #     color[color > 255] = 255

    #     distPos = (int(uniform() * (sampleSize[0] - distSize[0])),
    #                int(uniform() * (sampleSize[1] - distSize[1])))

    #     distractor = numPools[sampleNum][sampleIdx][1][distPos[0]:distPos[
    #         0] + distSize[0], distPos[1]:distPos[1] + distSize[1]]

    #     scale = float(uniform(scaleParam[0], scaleParam[1]))
    #     size = (int(distSize[0] * scale), int(distSize[1] * scale))

    #     pos = np.round(np.array([uniform(0, w - size[0]), uniform(0, h - size[1])]))
    #     pos = pos.astype(np.uint8)  # [height, width]

    #     distractor = imresize(distractor, size)
    #     background[pos[0]:pos[0] + size[0], pos[1]:pos[1] + size[1],
    #                0: 3] += (distractor[:, :, None] * color[None, None, :] / 255.).astype('uint8')
    #background[..., 3] = 255

    # put numbers
    for info in questionInfoList:

        sampleIdx = int(uniform() * len(numPools[info[0]]))

        # print mnistSample
        colorNoise = normal(colorParam[0], colorParam[1])
        color = info[2] + colorNoise
        color[color < 0] = 0
        color[color > 255] = 255

        while True:
            scale = float(uniform(scaleParam[0], scaleParam[1]))
            size = (int(sampleSize[0] * scale), int(sampleSize[1] * scale))

            #pos = np.round(np.array([uniform(0, w - size[0]), uniform(0, h - size[1])]))
            pos = np.round(np.array([uniform(0, h - size[1]), uniform(0, w - size[0])]))
            pos = pos.astype(np.int32)

            found = True
            for i in range(size[0]):
                for j in range(size[1]):
                    if (newImg[pos[0] + i, pos[1] + j, :] != 0).any():
                        found = False
                        break
                if not found:
                    break
            if found:
                break

        scales.append(scale)
        mnistSample = imresize(numPools[info[0]][sampleIdx][1], scale)

        box_coordinates = list()
        for i in range(size[0]):
            for j in range(size[1]):
                if mnistSample[i, j] != 0:
                    newImg[pos[0] + i, pos[1] + j, 0: 3] = color * (mnistSample[i, j]) / 255.
                    newImg[pos[0] + i, pos[1] + j, 3] = mnistSample[i, j]

        [ymin, xmin, ymax, xmax] = [pos[0], pos[1], pos[0] + size[0], pos[1] + size[1]]
        generated_bboxs.append([ymin + 2, xmin + 2, ymax - 2, xmax - 2])

    newImg[newImg > 255] = 255
    newImg[newImg < 0] = 0
    newImg = newImg.astype(np.uint8)

    img_ = image.fromarray(newImg)
    #img = image.fromarray(background)
    img = background
    background.paste(img_, None, img_)
    img = img.convert('RGB')

    ori_img_path = dest_path + '/imgs/%05d.png' % imgIdx
    bboxs_img_path = dest_path + '/imgs_bbox/%05d.png' % imgIdx

    img.save(ori_img_path)

    for coor in generated_bboxs:
        img_obj = ImageDraw.Draw(img)
        img_obj.rectangle((coor[1], coor[0], coor[3], coor[2]), outline="red")

    img.save(bboxs_img_path)

    return generated_bboxs


labels = []
if not os.path.exists(dest_path):
    os.makedirs(dest_path)
if not os.path.exists(dest_path + '/imgs'):
    os.makedirs(dest_path + '/imgs')
if not os.path.exists(dest_path + '/imgs_bbox'):
    os.makedirs(dest_path + '/imgs_bbox')

for i in range(n_syn_imgs):
    print('\r', i, end=' ')
    sys.stdout.flush()
    qInfo = generateSimpleColorQuestionInfo(minnum=15, maxnum=20)
    g_bboxs = renderImage(dest_path, i, qInfo, scaleParam=(0.7, 2.5))
    for idx, item in enumerate(qInfo):
        labels.append(','.join(['%05d' % i]
                               + [str(x) for x in item[:-1]]
                               + [str(coors) for coors in g_bboxs[idx]]))

labelOut = open(dest_path + '/labels.txt', 'wt')
labelOut.write('\n'.join(labels))
labelOut.close()
scaleOut = open(dest_path + '/scales.txt', 'wt')
scaleOut.write('\n'.join([str(x) for x in scales]))
scaleOut.close()
print()

generate_query_patch()
