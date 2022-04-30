# python libraties
import os,cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image


#print(os.listdir("input"))
# ## Step 1. Data analysis and preprocessing
# Get the all image data paths， match the row information in HAM10000_metadata.csv with its corresponding image
image_dir = '../Downloads/CUB_200_2011/images'
labels = '../Downloads/CUB_200_2011/images.txt'
bounding_boxes = '../Downloads/CUB_200_2011/bounding_boxes.txt'
train_test = '../Downloads/CUB_200_2011/train_test_split.txt'
cropped_train_path = './datasets/cub200_cropped/train_cropped/'
cropped_test_path = './datasets/cub200_cropped/test_cropped/'
label_dict = {}
bounding_box_dict = {}
train_test_dict = {}


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

label_text = open(labels)
lines = label_text.readlines()
for line in lines:
    split_line = line.split()
    label_dict[split_line[0]] = split_line[1]

bounding_boxes_text = open(bounding_boxes)
lines = bounding_boxes_text.readlines()
for line in lines:
    split_line = line.split()
    area = split_line[1:5]
    bounding_box_dict[split_line[0]] = area

train_test_text = open(train_test)
lines = train_test_text.readlines()
for line in lines:
    split_line = line.split()
    train_test_dict[split_line[0]] = split_line[1]


for key, value in label_dict.items():
    #print(f"{key}: {value}")
    folder= value.split("/")[0]
    image = value.split("/")[1]

    area = tuple(int(float(e)) for e in bounding_box_dict[key])
    area = (area[0],area[1],area[0]+area[2],area[1]+ area[3])
    #print(f"{area}")
    image_path = os.path.join(image_dir, value)
    img = Image.open(open(image_path, 'rb'))
    #area = (20, 20, 50, 50)
    cropped_img = img.crop(area)
    if int(train_test_dict[key])==1:
        #print(f"{key}: {train_test_dict[key]}")
        target_dir = cropped_train_path + folder +'/'
        makedir(target_dir)
        cropped_img.save(target_dir+image+".png","PNG")
    else:
        target_dir = cropped_test_path + folder +'/'
        makedir(target_dir)
        cropped_img.save(target_dir+image+".png","PNG")
    #cropped_img.show()

    #print(image_path)
#imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}
