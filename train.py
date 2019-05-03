import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable

# from net import vgg16, vgg16_bn
#from resnet_yolo import resnet50, resnet18
from yololoss import yoloLoss
from dataset import yoloDataset
from load_dataset import VotTrainDataset
#from visualize import Visualizer
import numpy as np
from models import Yolov1_vgg16bn
from resnet_model import resnet18
from resnet_model import resnet50
from resnet_model import resnet101
use_gpu = torch.cuda.is_available()

file_root = '/home/xiec/DLCV/hw2-xiechen0692/hw2_train_val/'
#learning_rate = 0.002
learning_rate = 0.002
num_epochs = 50
batch_size = 8


net = resnet50()
resnet = models.resnet50(pretrained=True)
new_state_dict = resnet.state_dict()
dd = net.state_dict()
for k in new_state_dict.keys():
    print(k)
    if k in dd.keys() and not k.startswith('fc'):
        print('yes')
        dd[k] = new_state_dict[k]
net.load_state_dict(dd)
criterion = yoloLoss(7, 2, 5, 0.5)
if use_gpu:
    net.cuda()

net.train()
# different learning rate
params = []
params_dict = dict(net.named_parameters())
for key, value in params_dict.items():
    if key.startswith('features'):
        params += [{'params': [value], 'lr': learning_rate * 1}]
    else:
        params += [{'params': [value], 'lr': learning_rate}]
optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)

train_dataset = yoloDataset(root=file_root,img_size=448,  transforms=[transforms.ToTensor()],train= True)#S=7, B=2, C=16,
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = yoloDataset(root=file_root, img_size=448, transforms=[transforms.ToTensor()],train=False)#S=7, B=2, C=16,
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
##
print('the dataset has %d images' % (len(train_dataset)))
print('the batch_size is %d' % (batch_size))
logfile = open('log.txt', 'w')

num_iter = 0
# vis = Visualizer(env='xiong')
best_test_loss = np.inf

for epoch in range(num_epochs):
    net.train()
    if epoch == 10:
        learning_rate = 0.0015
    if epoch == 20:
        learning_rate = 0.001
    if epoch == 50:
        learning_rate = 0.0005
    if epoch == 60:
        learning_rate = 0.0001
    if epoch == 80:
        learning_rate = 0.00001

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
    print('Learning Rate for this epoch: {}'.format(learning_rate))

    total_loss = 0.

    for i, (images, target) in enumerate(train_loader):
        images = Variable(images)
        target = Variable(target)
        if use_gpu:
            images, target = images.cuda(), target.cuda()
        # print("****",images.size(),target.size())
        pred = net(images)
        # print("####",pred.size())
        loss = criterion(pred, target)
        total_loss += loss.data#[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 5 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_loader), loss.data, total_loss / (i + 1)))#[0]
            num_iter += 1
            loss_train = total_loss / (i + 1)
            print(loss_train)
            #vis.plot_train_val(loss_train=total_loss / (i + 1))

    # validation
    validation_loss = 0.0
    net.eval()
    for i, (images, target) in enumerate(test_loader):
        images = Variable(images, requires_grad=False)
        target = Variable(target, requires_grad=False)
        if use_gpu:
            images, target = images.cuda(), target.cuda()

        pred = net(images)
        loss = criterion(pred, target)
        validation_loss += loss.data
    validation_loss /= len(test_loader)
    print("validation_loss", validation_loss)
    #vis.plot_train_val(loss_val=validation_loss)

    if best_test_loss > validation_loss:
        best_test_loss = validation_loss
        print('get best test loss %.5f' % best_test_loss)
        torch.save(net.state_dict(), 'best.pth')
    logfile.writelines(str(epoch) + '\t' + str(validation_loss) + '\n')
    logfile.flush()
    torch.save(net.state_dict(), 'yolo.pth')
    if epoch % 10 == 0:
        torch.save(net.state_dict(), 'yolo_{}epoch.pth'.format(epoch))