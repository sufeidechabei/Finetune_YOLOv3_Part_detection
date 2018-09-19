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
    mypath = './yolodata/' + dir + '/test/test.txt'
    imgpath =  './yolodata/' + dir + '/test/img/'
    print(os.listdir(imgpath))
    file_list = [f for f in os.listdir(imgpath)]
    lengthfile = len(file_list)
    if os.path.exists('./yolodata/' + dir + '/test/labels'):
       shutil.rmtree('./yolodata/' + dir + '/test/labels')
    if os.path.exists('./yolodata/' + dir + '/test/indextest.txt'):
        os.remove('./yolodata/' + dir + '/test/indextest.txt')
    os.makedirs('./yolodata/' + dir + '/test/labels')
    cls = "object"
    cls_id = classes.index(cls) + 1
    with open(mypath, 'r') as f:
        countline=1
        lines = f.readlines()
        cls_id = classes.index(cls) + 1
        for line in lines[:lengthfile]:
            line = line.strip()
            elems = line.split(' ')
            print(elems)
            print(len(elems))
            elems_len = len(elems)
            bounding_num = int(elems_len/5)
            image = Image.open(imgpath + '{:0>5d}.jpg'.format(countline))
            w = image.size[0]
            h = image.size[1]
            size = (w, h)
            bb_list = []
            for i in  range(bounding_num):
                xmin = elems[i*5 + 1]
                xmax = elems[i*5 + 2]
                ymin = elems[i*5 + 3]
                ymax = elems[i*5 + 4]
                box = (float(xmin), float(xmax), float(ymin), float(ymax))
                bb = convert(size, box)
                bb_list.append(bb)
            with open("./yolodata/" + dir + '/test/labels/{:0>5d}.txt'.format(countline), 'a+') as f:
                datawrite = ''
                for bb in bb_list:
                    f.write('1 ')
                    f.write(' '.join([str(a) for a in bb]))
                    f.write(' ')
                f.write('\r\n')
            with open('./yolodata/' + dir + '/test/indextest.txt', 'a+') as f:
                f.write(imgpath + '{:0>5d}.jpg\r\n'.format(countline) )
            countline = countline + 1















