from __future__ import division

from models import *
from models import Darknet
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from logger import Logger
log = Logger('./lossvisual')
os.environ["CUDA_VISIBLE_DEVICES"]="1"

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=30
                    , help='number of epochs')
parser.add_argument('--image_folder', type=str, default='data/samples', help='path to dataset')
parser.add_argument('--batch_size', type=int, default=16, help='size of each image batch')
parser.add_argument('--model_config_path', type=str, default='config/yolov3.cfg', help='path to model config file')
parser.add_argument('--data_config_path', type=str, default='config/coco.data', help='path to data config file')
parser.add_argument('--weights_path', type=str, default='checkpoints/yolov3.weights', help='path to weights file')
parser.add_argument('--class_path', type=str, default='data/coco.names', help='path to class label file')
parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0, help='iou thresshold for non-maximum suppression')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
parser.add_argument('--checkpoint_interval', type=int, default=1, help='interval between saving model weights')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='directory where model checkpoints are saved')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

os.makedirs('output', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)


# Get data configuration
data_config     = parse_data_config(opt.data_config_path)
# Get hyper parameters
hyperparams     = parse_model_config(opt.model_config_path)[0]
learning_rate   = float(hyperparams['learning_rate'])
momentum        = float(hyperparams['momentum'])
decay           = float(hyperparams['decay'])
burn_in         = int(hyperparams['burn_in'])

# Initiate model

dirlist = os.listdir('./yolodata/')
dirlist.remove('showTruth.m')
for dir in sorted(dirlist):
    # Get dataloader
    model = Darknet(opt.model_config_path)
    model.load_weights(opt.weights_path)

    if cuda:
        model = model.cuda()

    model.train()
    train_path = './yolodata/' + dir + '/train/indextrain.txt'
    dataloader = torch.utils.data.DataLoader(
        ListDataset(train_path),
        batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, dampening=0, weight_decay=decay)

    for epoch in range(opt.epochs):
        loss_list = []
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            imgs = Variable(imgs.type(Tensor))
            print(imgs.size())
            targets = Variable(targets.type(Tensor), requires_grad=False)
            print(targets.size())
            optimizer.zero_grad()

            loss = model(imgs, targets)
            loss_data = loss.item()
            loss.backward()
            loss_list.append(loss_data)
            optimizer.step()

            print(
                '[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f]' %
                (epoch, opt.epochs, batch_i, len(dataloader),
                 model.losses['x'], model.losses['y'], model.losses['w'],
                 model.losses['h'], model.losses['conf'], model.losses['cls'],
                 loss.item(), model.losses['recall']))

            model.seen += imgs.size(0)
        loss_result = sum(loss_list) / len(loss_list)
        log.scalar_summary('loss', loss_result, epoch + 1)

        if epoch % opt.checkpoint_interval == 0:
            model.save_weights('%s/%d.weights' % (opt.checkpoint_dir, epoch))
    model.eval()

    # Get dataloader
    test_path = './yolodata/' + dir + '/test/indextest.txt'
    dataset = ListDataset(test_path)
    dataset.max_objects = 9
    opt.batch_size = 1
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=opt.batch_size, shuffle=False, num_workers=1)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    n_gt = 0
    correct = 0

    print('Compute mAP...')

    outputs = []
    targets = None
    APs = []
    count = 0
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        imgs = Variable(imgs.type(Tensor))
        targets = targets.type(Tensor)

        with torch.no_grad():
            output = model(imgs)
            data = output[:, :, 5]
            _, index = torch.max(data, dim=1)
            out = output[0, index][:4]
            output = [out]
            for i in range(9):
                if targets[0, i, 0] == 0:
                    continue

                iou = bbox_iou(output[0][:, :4], targets[0, i, 1:5].unsqueeze(0) * imgs.size(3), False)
                print(iou)
                if iou > 0.5:
                    count = count + 1
                    break;
    testlength = os.listdir('./yolodata/' + dir +'/test/img/')
    print('The accuracy is: ' + str(count / len(testlength)))
    with open("./bigresult.txt", 'a+') as f:
        f.write(dir + ": " + str(count/len(testlength))+'\r\n')






