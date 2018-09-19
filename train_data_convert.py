import os
from os import walk, getcwd
from PIL import Image
import shutil


classes = ["object"]


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


dir_list = os.listdir('./yolodata/')
dir_list.remove('showTruth.m')
for dir in dir_list:
    mypath = './yolodata/' + dir + '/train/train.txt'
    imagepath = './yolodata/' + dir + '/train/img/'
    if os.path.exists('./yolodata/' + dir + '/train/labels'):
       shutil.rmtree('./yolodata/' + dir + '/train/labels')
    if os.path.exists('./yolodata/' + dir + '/train/indextrain.txt'):
        os.remove('./yolodata/' + dir + '/train/indextrain.txt')
    os.makedirs('./yolodata/' + dir + '/train/labels')
    cls = "object"
    if cls not in classes:
        exit(0)
    cls_id = classes.index(cls) + 1
    with open(mypath, 'r') as f:
        lines = f.readlines()
        print(len(lines))
        countline = 1
        for line in lines:
            line = line.strip()
            elems = line.split(' ')
            xmin = elems[0]
            xmax = elems[1]
            ymin = elems[2]
            ymax = elems[3]
            box = (float(xmin), float(xmax), float(ymin), float(ymax))
            image = Image.open(imagepath + '{:0>5d}.jpg'.format(countline))
            w = image.size[0]
            h = image.size[1]
            size = (w, h)
            bb = convert(size, box)


            with open('./yolodata/'+dir+'/train/labels/{:0>5d}.txt'.format(countline), 'a+') as f:
                f.write(str(cls_id) + ' ' + ' '.join([str(a) for a in bb]) + '\r\n')
            with open('./yolodata/' + dir + '/train/indextrain.txt', 'a+') as f:
                f.write('./yolodata/' + dir + '/train/img/{:0>5d}.jpg\r\n'.format(countline))
            countline = countline + 1





