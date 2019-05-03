#encoding:utf-8
#
#created by xiongzihua
#

import sys
import os

import cv2
import numpy as np

DOTA_CLASSES = (  # always index 0
    'plane', 'ship', 'storage-tank', 'baseball-diamond',
    'tennis-court', 'basketball-court', 'ground-track-field',
    'harbor', 'bridge', 'small-vehicle', 'large-vehicle',
    'helicopter', 'roundabout', 'soccer-ball-field',
    'swimming-pool', 'container-crane')

Color = [[0, 0, 0],
                    [128, 0, 0],
                    [0, 128, 0],
                    [128, 128, 0],
                    [0, 0, 128],
                    [128, 0, 128],
                    [0, 128, 128],
                    [128, 128, 128],
                    [64, 0, 0],
                    [192, 0, 0],
                    [64, 128, 0],
                    [192, 128, 0],
                    [64, 0, 128],
                    [192, 0, 128],
                    [64, 128, 128],
                    [192, 128, 128],
                    [0, 64, 0],
                    [128, 64, 0],
                    [0, 192, 0],
                    [128, 192, 0],
                    [0, 64, 128]]
        
def parse_det(detfile):
    result = []
    with open(detfile, 'r') as f:
        for line in f:
            token = line.strip().split()
            if len(token) != 10:
                continue
            x1 = int(float(token[0]))
            y1 = int(float(token[1]))
            x2 = int(float(token[4]))
            y2 = int(float(token[5]))
            cls = token[8]
            prob = float(token[9])
            result.append([(x1,y1),(x2,y2),cls,prob])
    return result 

if __name__ == '__main__':
    imgfile = sys.argv[1]
    detfile = sys.argv[2]

    image = cv2.imread(imgfile)
    result = parse_det(detfile)
    for left_up,right_bottom,class_name,prob in result:
        color = Color[DOTA_CLASSES.index(class_name)]
        cv2.rectangle(image,left_up,right_bottom,color,2)
        label = class_name+str(round(prob,2))
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        p1 = (left_up[0], left_up[1]- text_size[1])
        cv2.rectangle(image, (p1[0] - 2//2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
        cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)

    cv2.imwrite('result.jpg',image)

