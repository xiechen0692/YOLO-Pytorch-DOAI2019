# encoding:utf-8
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
from models import Yolov1_vgg16bn
import sys

VOC_CLASSES = (
    'plane', 'ship', 'storage-tank', 'baseball-diamond',
    'tennis-court', 'basketball-court', 'ground-track-field',
    'harbor', 'bridge', 'small-vehicle', 'large-vehicle',
    'helicopter', 'roundabout', 'soccer-ball-field',
    'swimming-pool', 'container-crane')

def decoder(pred):
    grid_num = 7
    boxes = []
    cls_indexs = []
    probs = []
    cell_size = 1. / grid_num
    pred = pred.data
    pred = pred.squeeze(0)  # 7x7x30
    contain1 = pred[:, :, 4].unsqueeze(2)
    contain2 = pred[:, :, 9].unsqueeze(2)
    contain = torch.cat((contain1, contain2), 2)
    mask1 = contain > 0.3  # 大于阈值
    mask2 = (contain == contain.max())  # we always select the best contain_prob what ever it>0.9
    mask = (mask1 + mask2).gt(0)
    # min_score,min_index = torch.min(contain,2) #每个cell只选最大概率的那个预测框
    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                # index = min_index[i,j]
                # mask[i,j,index] = 0
                if mask[i, j, b] == 1:
                    # print(i,j,b)
                    box = pred[i, j, b * 5:b * 5 + 4]
                    contain_prob = torch.FloatTensor([pred[i, j, b * 5 + 4]])
                    xy = torch.FloatTensor([j, i]) * cell_size  # cell左上角  up left of cell
                    box[:2] = box[:2] * cell_size + xy  # return cxcy relative to image
                    box_xy = torch.FloatTensor(box.size())  # 转换成xy形式    convert[cx,cy,w,h] to [x1,xy1,x2,y2]
                    box_xy[:2] = box[:2] - 0.5 * box[2:]
                    box_xy[2:] = box[:2] + 0.5 * box[2:]
                    max_prob, cls_index = torch.max(pred[i, j, 10:], 0)
                    if float((contain_prob * max_prob)[0]) > 0.1:
                        boxes.append(box_xy.view(1, 4))
                        cls_indexs.append(cls_index)
                        probs.append(contain_prob * max_prob)
    if len(boxes) == 0:
        boxes = torch.zeros((1, 4))
        probs = torch.zeros(1)
        cls_indexs = torch.zeros(1)
    else:
        boxes = torch.cat(boxes, 0)  # (n,4)
        probs = torch.cat(probs, 0)  # (n,)
        cls_indexs = torch.stack(cls_indexs, 0)  # (n,)
    # print(boxes.size(), probs.size())
    keep = nms(boxes, probs)
    return boxes[keep], cls_indexs[keep], probs[keep]


def nms(bboxes, scores, threshold=0.3):
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    # print(x2-x1)

    _, order = scores.sort(0, descending=True)
    keep = []
    while order.numel() > 0:

        if order.numel() == 1:
            keep.append(order.data.item())
            break

        i = order.data[0].item()
        keep.append(i)

        # print(x1[i])
        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr <= threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]
    return torch.LongTensor(keep)


def predict_output(model, image_name, root_path=''):
    result = []
    image = cv2.imread(root_path + image_name)
    h, w, _ = image.shape
    img = cv2.resize(image, (448, 448))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean = (123, 117, 104)  # RGB
    img = img - np.array(mean, dtype=np.float32)

    transform = transforms.Compose([transforms.ToTensor(), ])
    img = transform(img)
    img = Variable(img[None, :, :, :], requires_grad=False)
    img = img.cuda()

    pred = model(img)  # 1x7x7x30
    pred = pred.cpu()
    boxes, cls_indexs, probs = decoder(pred)

    for i, box in enumerate(boxes):
        x1 = int(box[0] * w)
        x2 = int(box[2] * w)
        y1 = int(box[1] * h)
        y2 = int(box[3] * h)
        cls_index = cls_indexs[i]
        cls_index = int(cls_index)  # convert LongTensor to int
        prob = probs[i]
        prob = float(prob)
        result.append([(x1, y1), (x2, y2), VOC_CLASSES[cls_index], image_name, prob])
    return result

def transform_result(result):
    xmin = float(result[0][0])
    ymin = float(result[0][1])
    xmax = float(result[1][0])
    ymax = float(result[1][1])
    #xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax, label
    x1 = xmin
    y1 = ymin  # left-up
    x2 = xmax
    y2 = ymin  # right-up
    x3 = xmax
    y3 = ymax  # right-down
    x4 = xmin
    y4 = ymax  # leftdown
    lebel = result[2]
    my_list = []
    my_list.append(x1)
    my_list.append(y1)
    my_list.append(x2)
    my_list.append(y2)
    my_list.append(x3)
    my_list.append(y3)
    my_list.append(x4)
    my_list.append(y4)
    my_list.append(lebel)
    my_list.append(0)
    return " ".join(str(item) for item in my_list)

if __name__ == '__main__':

    model_path = 'vgg16_80epoch.pth'
    testing_images = sys.argv[1]
    output_prediction = sys.argv[2]

    model = Yolov1_vgg16bn(pretrained=False)
    print('#load model#')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.cuda()

    print('#predicting#')
    files = os.listdir(testing_images)
    files.sort()
    for file in files:
        text_file = open('%s%s' % (output_prediction,(file.split('.')[0]+'.txt')), 'w')#creat a new txt according to the image file
        #os.mknod(save_file + "{}.txt".format(file.split('.')[0]))
        results = predict_output(model, file, root_path=testing_images)
        for result in results:
            # print(transform_result(result))
            text_file.write(transform_result(result)+'\n')
        text_file.close()
