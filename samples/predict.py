
# coding: utf-8

# # Mask R-CNN Demo
# 
# A quick intro to using the pre-trained model to detect and segment objects.

# In[1]:


import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

#get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


# ## Configurations
# 
# We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the ```CocoConfig``` class in ```coco.py```.
# 
# For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.

# In[2]:


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# ## Create Model and Load Trained Weights

# In[3]:


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# ## Class Names
# 
# The model classifies objects and returns class IDs, which are integer value that identify each class. Some datasets assign integer values to their classes and some don't. For example, in the MS-COCO dataset, the 'person' class is 1 and 'teddy bear' is 88. The IDs are often sequential, but not always. The COCO dataset, for example, has classes associated with class IDs 70 and 72, but not 71.
# 
# To improve consistency, and to support training on data from multiple sources at the same time, our ```Dataset``` class assigns it's own sequential integer IDs to each class. For example, if you load the COCO dataset using our ```Dataset``` class, the 'person' class would get class ID = 1 (just like COCO) and the 'teddy bear' class is 78 (different from COCO). Keep that in mind when mapping class IDs to class names.
# 
# To get the list of class names, you'd load the dataset and then use the ```class_names``` property like this.
# ```
# # Load COCO dataset
# dataset = coco.CocoDataset()
# dataset.load_coco(COCO_DIR, "train")
# dataset.prepare()
# 
# # Print class names
# print(dataset.class_names)
# ```
# 
# We don't want to require you to download the COCO dataset just to run this demo, so we're including the list of class names below. The index of the class name in the list represent its ID (first class is 0, second is 1, third is 2, ...etc.)

# In[4]:


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
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


# ## Run Object Detection

# In[5]:

mask_output_dir = "masked_neg_images"

def read_urls():
    urls=[]
    f = open("images.txt")               # 返回一个文件对象
    line = f.readline()               # 调用文件的 readline()方法
    while line:
        urls.append(line.replace('\n',''))
        line = f.readline()
    f.close()
    return urls

image_names = read_urls()

def maxRoi(rois):
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
image_size = len(image_names)
start_idx = 0
for image_idx in range(start_idx,image_size,1):
    img_name = image_names[image_idx]
    masked_image_name = mask_output_dir+"/"+img_name.split('/')[1]
    if os.path.exists(masked_image_name) == True:
        print("detected image ",img_name)
        continue
    print("begin detect ",img_name,"image idx=",image_idx)
    if os.path.exists(img_name) == False:
        print(img_name+" not exists")
        continue

    image = cv2.imread(img_name)
    # Run detection
    results = []
    try:
        results = model.detect([image], verbose=1)
    except:
        print("exception image",img_name)
    print("finish detect ", img_name)
    if len(results) == 0:
        print("no obj is detected for img",img_name)
        continue
    r = results[0]
    #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    none_zero = 0
    max_roi, idx = maxRoi(r['rois'])
    if idx == -1:
        print("no person detect in image ", img_name)
        continue
    for i in range(max_roi[0],max_roi[2]):
        for j in range(max_roi[1],max_roi[3]):
            if r['masks'][i][j][idx] == False:
                image[i][j][0]=255
                image[i][j][1]=255
                image[i][j][2]=255
    newimage = image[max_roi[0]:max_roi[2],max_roi[1]:max_roi[3]]
    cv2.imwrite(masked_image_name, newimage)
    print("finish write mask image",masked_image_name)
    

#i_len = results[0]['masks'].shape[0]
#j_len = results[0]['masks'].shape[1]
#k_len = results[0]['masks'].shape[2]
#print("i_len",i_len,"j_len",j_len,"k_len",k_len)
#print(image.shape)
#print(r['rois'])
#print(r['class_ids'])
#print("r['masks'][0][0][0]",r['masks'][0][0][0] == False)



'''
for i in range(i_len):
    for j in range(j_len):
        if r['masks'][i][j][idx] == False:
            image[i][j][0]=255
            image[i][j][1]=255
            image[i][j][2]=255
'''

'''
for i in range(i_len):
    for j in range(j_len):
        flag = False
        for k in range(k_len):
            if r['class_ids'][k] == 1:
                flag = flag or r['masks'][i][j][k]
        if flag == False:
            image[i][j][0]=255
            image[i][j][1]=255
            image[i][j][2]=255
'''


