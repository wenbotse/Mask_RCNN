
# coding: utf-8

from keras.layers import BatchNormalization
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import np_utils
from keras.callbacks import TensorBoard

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import hashlib
import requests
from download import safe_download
import traceback


ROOT_DIR = os.path.abspath("../")

sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

model.load_weights(COCO_MODEL_PATH, by_name=True)



class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


def result_callback(url, color_prob, gender_prob):
    print("call back set url=",url," prob=", color_prob," gender_prob=",gender_prob)
    file = "detect_result.txt"
    with open(file, 'a+') as f:
        f.write(url + "\t" + str(color_prob) +"\t"+str(gender_prob)+ "\n")
    f.close()

def load_image(image_url):
    image_data_list = []
    image = load_img(image_url, target_size=(128, 128))
    image_data_list.append(img_to_array(image))
    img_data = np.array(image_data_list)
    img_data = img_data.astype('float32')
    img_data /= 255
    return img_data

def make_gender_network():
    model = Sequential()
    model.add(Conv2D(32, (3, 3),activation='relu', padding='same', input_shape=(128, 128, 3)))
    model.add(Conv2D(64, (3, 3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128,(2,2),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid'))
    model.summary()
    return model

def make_color_network():
    model = Sequential()
    model.add(Conv2D(48, (5, 5), strides=2, padding='valid', input_shape=(128, 128, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2),strides=2))


    model.add(Conv2D(128, (3, 3), padding='valid'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.5))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(Flatten())

    model.add(Dense(2048))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2048))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    #model.summary()
    return model

color_classify_model = make_color_network()
#color_classify_model.load_weights("./facerank.h5.5_0.49827")
#color_classify_model.load_weights("facerank.h5.0_0.51556")
#color_classify_model.load_weights("facerank.h5.6_0.37979")
color_classify_model.load_weights("facerank.h5.7_0.37896")

gender_classify_model = make_gender_network()
#gender_classify_model.load_weights("gender.h5.5_0.45093")
#gender_classify_model.load_weights("gender.h5.4_0.43926")
gender_classify_model.load_weights("gender.h5.8_0.37923")
def generate_urls(num=10):
    #arr = ["http://img.mxtrip.cn/fadd1b80f8f62eb335cca0a1ffb777f1.jpeg"]
    urls=[]
    f = open("urls.txt")               # 返回一个文件对象
    line = f.readline()               # 调用文件的 readline()方法
    while line:
        urls.append(line.replace('\n',''))
        if len(urls) == 10:
            print(line)
            yield urls
            urls=[]
        line = f.readline()
    f.close()
    yield urls

def maxRoi(rois, r):
    idx = -1
    max_area = 0
    for i in range(len(rois)):
         if r['class_ids'][i] == 1:
             x1 = rois[i][0]
             y1 = rois[i][1]
             x2 = rois[i][2]
             y2 = rois[i][3]
             area = (x2-x1)*(y2-y1)
             if area > max_area:
                 max_area = area
                 idx = i
    if idx == -1:
        return "", idx
    return rois[idx], idx 

def run():
    urls_gen =  generate_urls()
    for urls in urls_gen:
        for url in urls:
            run_detect(url)             

def run_detect(url):
    md5=hashlib.md5(url.encode('utf-8')).hexdigest()
    name = md5+'.jpg'
    img_name = "temp_image/"+name
    masked_image_name = "masked_image/"+name
    if os.path.exists("temp_image/"+name) == False :
        print("begin to download imgurl=",url)
        img_name = safe_download("temp_image",url)
    else:
        print("exist file name="+"temp_image/"+name) 
    print("begin detect ", img_name)
    if os.path.exists(img_name) == False:
        print(img_name+" not exists")
        return

    image = cv2.imread(img_name)
    results = []
    try:
        results = model.detect([image], verbose=1)
    except:
        print("exception image",img_name)
        return
    print("finish detect ", img_name)
    if len(results) == 0:
        print("no obj is detected for img",img_name)
        return
    r = results[0]
    #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    none_zero = 0
    max_roi, idx = maxRoi(r['rois'], r)
    if idx == -1:
        print("no person detect in image ", img_name)
        return
    for i in range(max_roi[0],max_roi[2]):
        for j in range(max_roi[1],max_roi[3]):
            if r['masks'][i][j][idx] == False:
                image[i][j][0]=255
                image[i][j][1]=255
                image[i][j][2]=255
    newimage = image[max_roi[0]:max_roi[2],max_roi[1]:max_roi[3]]
    cv2.imwrite(masked_image_name, newimage)
    #print("finish write mask image",masked_image_name)
    im=Image.open(masked_image_name)
    out = im.resize((128, 128))
    out.save("images_resize/"+name)
    image = load_image("images_resize/"+name)
    color_probs = color_classify_model.predict(image, verbose=0)
    gender_probs = gender_classify_model.predict(image, verbose=0)

    result_callback(url, color_probs[0][0], gender_probs[0][0])     
    if os.path.exists("images_resize/"+name) == True :
        os.remove("images_resize/"+name)
        #print("delete file name="+("images_resize/"+name))
    if os.path.exists("masked_image/"+name) == True :
        os.remove("masked_image/"+name)
        #print("delete file name="+("masked_image/"+name))
    if os.path.exists("temp_image/"+name) == True :
        os.remove("temp_image/"+name)
        #print("delete file name="+("temp_image/"+name))
if __name__ == '__main__':
    run()
