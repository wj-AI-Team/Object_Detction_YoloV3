from ctypes import *
import math
import random
import glob
import pandas as pd
import os
import cv2
import time
import csv

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("/home/alen/darknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    net = load_net("/home/alen/darknet/cfg/yolov3-voc.cfg".encode('utf-8'), "/home/alen/darknet/backup/yolov3-voc_final.weights".encode('utf-8'), 0)
    meta = load_meta("/home/alen/darknet/cfg/voc.data".encode('utf-8'))    

    # 测试数据集的路径
    test_dir='/home/alen/darknet/scripts/2019_test/'
    # 检测结果保存路径 1h220片段
    save_dir = '/home/alen/darknet/result/'

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    pics = os.listdir(test_dir)
    count = 0
    res_result=[]
    for im in pics[:10]:
        img = os.path.join(test_dir, im)
        s = time.time()
        r = detect(net, meta, img.encode('utf-8'))
        # 输出的检测结果中坐标信息为目标的中心点坐标和box的w和w
                # print("一张图检测耗时：%.3f秒" % (time.time() - s))
        im = cv2.imread(img)
        for res in r:
            #print(type(res),dir(res))
            print(res,res[0].decode('utf-8'))
            x1 = int(res[2][0] - (res[2][2] / 2))
            y1 = int(res[2][1] - (res[2][3] / 2))
            x2 = x1 + int(res[2][2])
            y2 = y1 + int(res[2][3])
            cv2.rectangle(im, (x1-5, y1-5), (x2+5, y2+5), (0, 255, 0), 2)
            cv2.putText(im, str(res[0]).split("'")[1], (x1-10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imwrite(save_dir +str(count) +'.png', im)
            res_result+=[res[0].decode('utf-8')]
        count += 1
#columns={'test_unmber','data_name'}
#df_test=pd.DataFrame({'test_unmber':res_result,'data_name':res_result})
#df_test.to_csv('Alen.csv')
#print(res_result)
#print(df_test)


# Done
# df = pd.DataFrame(save_for_confusion_matrix)
# df.to_csv('your_filename.csv')
    
    

