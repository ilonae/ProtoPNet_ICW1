from tqdm import tqdm
import os,glob, random, cv2
import imageio
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms

from sklearn.model_selection import train_test_split

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def compute_img_mean_std(image_paths):
    """
        computing the mean and std of three channel on the whole dataset,
        first we should normalize the image from 0-255 to 0-1
    """

    img_h, img_w = 224, 224
    imgs = []
    means, stdevs = [], []

    for i in tqdm(range(len(image_paths))):
        img = cv2.imread(image_paths[i])
        img = cv2.resize(img, (img_h, img_w))
        imgs.append(img)

    imgs = np.stack(imgs, axis=3)
    print(imgs.shape)

    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # resize to one row
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse()  # BGR --> RGB
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    return means,stdevs


data_dir = '../LRP_pruning/data/ham1000'
dest_path = './datasets/ham10000/'
#print(os.listdir(data_dir))
all_image_path = glob.glob(os.path.join(data_dir, '*', '*.jpg'))
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'dermatofibroma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

#print(norm_mean, norm_std)
df_original = pd.read_csv(os.path.join(data_dir, 'HAM10000_metadata.csv'))
df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)
df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes
#print(df_original.head())
unique_types = pd.unique(df_original["cell_type"])

for unique_type in unique_types:
    makedir(dest_path + "/"+unique_type)

df = df_original.reset_index()  # make sure indexes pair with number of rows
for index, row in df.iterrows():
    name =row['path'].split("\\")[-1]
    name = name.split(".")[0]
    #print(name)
    im = cv2.imread(os.path.normpath(row['path']))
    cv2.imwrite(dest_path + "/"+row['cell_type']+"/"+name+".png", im)
                

