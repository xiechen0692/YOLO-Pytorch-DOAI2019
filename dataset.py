import os
import sys
import os.path

import random
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import cv2
import matplotlib.pyplot as plt


class yoloDataset(data.Dataset):
    image_size = 448

    def __init__(self, root, img_size, transforms , train):
        print('data init')
        self.train = train
        self.boxes = []
        self.labels = []
        self.mean = (123, 117, 104)  # RGB
        self.file_names = []
        self.img_size = img_size
        self.image_path = ""
        self.transform = transforms
        self.bboxes = []
        self.labels = []
        self.root = root

        VOC_CLASSES = (  # always index 0
            'plane', 'ship', 'storage-tank', 'baseball-diamond',
            'tennis-court', 'basketball-court', 'ground-track-field',
            'harbor', 'bridge', 'small-vehicle', 'large-vehicle',
            'helicopter', 'roundabout', 'soccer-ball-field',
            'swimming-pool', 'container-crane')
        ##
        #root = train15000 or val1500 and then add images or labelTxt_hbb
        if self.train:
            path = root + 'train15000' + '/' + 'labelTxt_hbb/'
            self.image_path = root + '/' + 'train15000' + '/' + 'images/'
        else:
            path = root + 'val1500' + '/' + 'labelTxt_hbb/'
            self.image_path =  root + '/' + 'val1500' + '/' + 'images/'

        files = os.listdir(path)
        files.sort()
        #print(files)
        for file in files:
            self.file_names.append(file.split('.')[0])
            with open(path + file) as f:
                lines = f.readlines()
            bbox = []
            label = []
            for line in lines:
                splited = line.strip().split()
                xmin = float(splited[0])
                ymin = float(splited[1])
                xmax = float(splited[2])
                ymax = float(splited[5])
                c = splited[8]
                # print([x, y, w, h])
                bbox.append([xmin, ymin, xmax, ymax])
                # print(c,self.dictset[c])
                # label.append(self.dictset[c])
                label.append(VOC_CLASSES.index(c))
            self.boxes.append(torch.Tensor(bbox))#at last len is 1500
            self.labels.append(torch.Tensor(label))#at last len is 1500 IntTensor
        self.num_samples = len(self.labels)#at last n_data is 1500
        #print(self.file_names)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.image_path + self.file_names[idx] + ".jpg"))
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()
        # print("idx:",idx)
        # print("labels:",labels)
        if self.train:
            img, boxes = self.random_flip(img, boxes)
            img, boxes = self.randomScale(img, boxes)
            img = self.randomBlur(img)
            img = self.RandomBrightness(img)
            img, boxes, labels = self.randomShift(img, boxes, labels)
            img, boxes, labels = self.randomCrop(img, boxes, labels)

        h, w, _ = img.shape
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)
        img = self.BGR2RGB(img)  # because pytorch pretrained model use RGB
        img = self.subMean(img, self.mean)  # 减去均值
        img = cv2.resize(img, (self.image_size, self.image_size))
        target = self.encode(boxes, labels)  # 7x7x26
        for t in self.transform:
            img = t(img)

        return img, target

    def __len__(self):
        return self.num_samples

    def encode(self, boxes, labels):
        grid = 7
        twoBandC = 26
        target = torch.zeros((grid, grid, twoBandC))
        cellsize = 1. / grid
        wh = boxes[:, 2:] - boxes[:, :2]
        centroid_xandy = (boxes[:, 2:] + boxes[:, :2]) / 2
        for i in range(centroid_xandy.size()[0]):
            xy_sample = centroid_xandy[i]
            ij = (xy_sample / cellsize).ceil() - 1
            if ij[1] > 6:
                ij[1] = 6
            if ij[0] > 6:
                ij[0] = 6
            X = int(ij[1])
            Y = int(ij[0])
            C = int(labels[i])
            xy = ij * cellsize 
            delta_xy = (xy_sample - xy) / cellsize
            target[X, Y, 2:4] = wh[i]
            target[X, Y, :2] = delta_xy
            target[X, Y, 7:9] = wh[i]
            target[X, Y, 5:7] = delta_xy
            target[X, Y, 4] = 1
            target[X, Y, 9] = 1
            target[X, Y, C + 10] = 1
        return target



    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def BGR2HSV(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def HSV2BGR(self, img):
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    def RandomBrightness(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            v = v * adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomSaturation(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            s = s * adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomHue(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            h = h * adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def randomBlur(self, bgr):
        if random.random() < 0.5:
            bgr = cv2.blur(bgr, (5, 5))
        return bgr

    def randomShift(self, bgr, boxes, labels):
        # 平移变换
        center = (boxes[:, 2:] + boxes[:, :2]) / 2
        if random.random() < 0.5:
            height, width, c = bgr.shape
            after_shfit_image = np.zeros((height, width, c), dtype=bgr.dtype)
            after_shfit_image[:, :, :] = (104, 117, 123)  # bgr
            shift_x = random.uniform(-width * 0.2, width * 0.2)
            shift_y = random.uniform(-height * 0.2, height * 0.2)
            # print(bgr.shape,shift_x,shift_y)
            # 原图像的平移
            if shift_x >= 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, int(shift_x):, :] = bgr[:height - int(shift_y), :width - int(shift_x),
                                                                     :]
            elif shift_x >= 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y), int(shift_x):, :] = bgr[-int(shift_y):, :width - int(shift_x),
                                                                              :]
            elif shift_x < 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, :width + int(shift_x), :] = bgr[:height - int(shift_y), -int(shift_x):,
                                                                             :]
            elif shift_x < 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y), :width + int(shift_x), :] = bgr[-int(shift_y):,
                                                                                      -int(shift_x):, :]

            shift_xy = torch.FloatTensor([[int(shift_x), int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:, 0] > 0) & (center[:, 0] < width)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < height)
            mask = (mask1 & mask2).view(-1, 1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if len(boxes_in) == 0:
                return bgr, boxes, labels
            box_shift = torch.FloatTensor([[int(shift_x), int(shift_y), int(shift_x), int(shift_y)]]).expand_as(
                boxes_in)
            boxes_in = boxes_in + box_shift
            labels_in = labels[mask.view(-1)]
            return after_shfit_image, boxes_in, labels_in
        return bgr, boxes, labels

    def randomScale(self, bgr, boxes):
        # 固定住高度，以0.8-1.2伸缩宽度，做图像形变
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            height, width, c = bgr.shape
            bgr = cv2.resize(bgr, (int(width * scale), height))
            scale_tensor = torch.FloatTensor([[scale, 1, scale, 1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
            return bgr, boxes
        return bgr, boxes

    def randomCrop(self, bgr, boxes, labels):
        if random.random() < 0.5:
            center = (boxes[:, 2:] + boxes[:, :2]) / 2
            height, width, c = bgr.shape
            h = random.uniform(0.6 * height, height)
            w = random.uniform(0.6 * width, width)
            x = random.uniform(0, width - w)
            y = random.uniform(0, height - h)
            x, y, h, w = int(x), int(y), int(h), int(w)

            center = center - torch.FloatTensor([[x, y]]).expand_as(center)
            mask1 = (center[:, 0] > 0) & (center[:, 0] < w)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < h)
            mask = (mask1 & mask2).view(-1, 1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if (len(boxes_in) == 0):
                return bgr, boxes, labels
            box_shift = torch.FloatTensor([[x, y, x, y]]).expand_as(boxes_in)

            boxes_in = boxes_in - box_shift
            boxes_in[:, 0] = boxes_in[:, 0].clamp_(min=0, max=w)
            boxes_in[:, 2] = boxes_in[:, 2].clamp_(min=0, max=w)
            boxes_in[:, 1] = boxes_in[:, 1].clamp_(min=0, max=h)
            boxes_in[:, 3] = boxes_in[:, 3].clamp_(min=0, max=h)

            labels_in = labels[mask.view(-1)]
            img_croped = bgr[y:y + h, x:x + w, :]
            return img_croped, boxes_in, labels_in
        return bgr, boxes, labels

    def subMean(self, bgr, mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr

    def random_flip(self, im, boxes):
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h, w, _ = im.shape
            xmin = w - boxes[:, 2]
            xmax = w - boxes[:, 0]
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
            return im_lr, boxes
        return im, boxes

    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            im = im * alpha + random.randrange(-delta, delta)
            im = im.clip(min=0, max=255).astype(np.uint8)
        return im



