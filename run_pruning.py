import os
import shutil

import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from PIL import Image
import os,glob, random, cv2
from tqdm import tqdm
import numpy as np
import pandas as pd

import argparse

from helpers import makedir
# load the data
from settings import train_dir, test_dir, train_push_dir
import model
import push
import prune
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function


class HAM10000(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        
        X = Image.open(self.df['path'][index])
        y = torch.tensor(int(self.df['cell_type_idx'][index]))

        if self.transform:
            X = self.transform(X)

        return X, y

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

    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # resize to one row
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse()  # BGR --> RGB
    stdevs.reverse()

    return means,stdevs


def get_ham1000():
    data_dir = './datasets/ham10000'
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
    norm_mean,norm_std = compute_img_mean_std(all_image_path)
    #print(norm_mean, norm_std)
    df_original = pd.read_csv(os.path.join(data_dir, 'HAM10000_metadata.csv'))
    df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
    df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)
    df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes
    #print(df_original.head())

    # this will tell us how many images are associated with each lesion_id
    df_undup = df_original.groupby('lesion_id').count()
    # now we filter out lesion_id's that have only one image associated with it
    df_undup = df_undup[df_undup['image_id'] == 1]
    df_undup.reset_index(inplace=True)
    #print(df_undup.head())
    # here we identify lesion_id's that have duplicate images and those that have only one image.
    def get_duplicates(x):
        unique_list = list(df_undup['lesion_id'])
        if x in unique_list:
            return 'unduplicated'
        else:
            return 'duplicated'

    # create a new colum that is a copy of the lesion_id column
    df_original['duplicates'] = df_original['lesion_id']
    # apply the function to this new column
    df_original['duplicates'] = df_original['duplicates'].apply(get_duplicates)
    df_original.head()
    df_original['duplicates'].value_counts()
    # now we filter out images that don't have duplicates
    df_undup = df_original[df_original['duplicates'] == 'unduplicated']
    #print(df_undup.shape)
    y = df_undup['cell_type_idx']
    _, df_test = train_test_split(df_undup, test_size=0.2, random_state=101, stratify=y)
    #print(df_val.shape)
    df_test['cell_type_idx'].value_counts()
    # This set will be df_original excluding all rows that are in the val set
    # This function identifies if an image is part of the train or val set.
    def get_val_rows(x):
        # create a list of all the lesion_id's in the val set
        val_list = list(df_test['image_id'])
        if str(x) in val_list:
            return 'val'
        else:
            return 'train'

    # identify train and val rows
    # create a new colum that is a copy of the image_id column
    df_original['train_or_val'] = df_original['image_id']
    # apply the function to this new column
    df_original['train_or_val'] = df_original['train_or_val'].apply(get_val_rows)
    # filter out train rows
    df_train = df_original[df_original['train_or_val'] == 'train']
    #print(len(df_train))
    #print(len(df_val))
    #print(df_val['cell_type'].value_counts())
    # Copy fewer class to balance the number of 7 classes
    data_aug_rate = [15,10,5,50,0,40,5]
    for i in range(7):
        if data_aug_rate[i]:
            df_train=df_train.append([df_train.loc[df_train['cell_type_idx'] == i,:]]*(data_aug_rate[i]-1), ignore_index=True)
    #print(df_train['cell_type'].value_counts())
    df_train, df_val = train_test_split(df_train, test_size=0.2)
    df_train = df_train.reset_index()
    df_val = df_val.reset_index()
    df_test = df_test.reset_index()

    train_transform = transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),transforms.RandomRotation(20),
                                      transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                        transforms.ToTensor()])
    # define the transformation of the val images.
    val_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                                        transforms.Normalize(norm_mean, norm_std)])
    training_set = HAM10000(df_train, transform=val_transform)
    validation_set = HAM10000(df_val, transform=train_transform)
    test_set = HAM10000(df_test, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(training_set, batch_size=32, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(validation_set, batch_size=32, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4)

    return train_loader, val_loader,test_loader



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpuid', nargs=1, type=str, default='0')
    parser.add_argument('-modeldir', nargs=1, type=str)
    parser.add_argument('-model', nargs=1, type=str)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]


    optimize_last_layer = True

    # pruning parameters
    k = 6
    prune_threshold = 3

    original_model_dir = args.modeldir[0] #'./saved_models/densenet161/003/'
    original_model_name = args.model[0] #'10_16push0.8007.pth'

    need_push = ('nopush' in original_model_name)
    if need_push:
        assert(False) # pruning must happen after push
    else:
        epoch = original_model_name.split('push')[0]

    if '_' in epoch:
        epoch = int(epoch.split('_')[0])
    else:
        epoch = int(epoch)

    model_dir = os.path.join(original_model_dir, 'pruned_prototypes_epoch{}_k{}_pt{}'.format(epoch,
                                            k,
                                            prune_threshold))
    makedir(model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)

    log, logclose = create_logger(log_filename=os.path.join(model_dir, 'prune.log'))

    ppnet = torch.load(original_model_dir + original_model_name)
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)
    class_specific = True

    train_loader, train_push_loader, test_loader = get_ham1000()

    log('training set size: {0}'.format(len(train_loader.dataset)))
    log('test set size: {0}'.format(len(test_loader.dataset)))
    log('push set size: {0}'.format(len(train_push_loader.dataset)))

    tnt.test(model=ppnet_multi, dataloader=test_loader,
            class_specific=class_specific, log=log)

    # prune prototypes
    log('prune')
    prune.prune_prototypes(dataloader=train_push_loader,
                        prototype_network_parallel=ppnet_multi,
                        k=k,
                        prune_threshold=prune_threshold,
                        preprocess_input_function=preprocess_input_function, # normalize
                        original_model_dir=original_model_dir,
                        epoch_number=epoch,
                        #model_name=None,
                        log=log,
                        copy_prototype_imgs=True)
    accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                    class_specific=class_specific, log=log)
    save.save_model_w_condition(model=ppnet, model_dir=model_dir,
                                model_name=original_model_name.split('push')[0] + 'prune',
                                accu=accu,
                                target_accu=0.70, log=log)

    # last layer optimization
    if optimize_last_layer:
        last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': 1e-4}]
        last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

        coefs = {
            'crs_ent': 1,
            'clst': 0.8,
            'sep': -0.08,
            'l1': 1e-4,
        }

        log('optimize last layer')
        tnt.last_only(model=ppnet_multi, log=log)
        for i in range(100):
            log('iteration: \t{0}'.format(i))
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                        class_specific=class_specific, coefs=coefs, log=log)
            accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                            class_specific=class_specific, log=log)
            save.save_model_w_condition(model=ppnet, model_dir=model_dir,
                                        model_name=original_model_name.split('push')[0] + '_' + str(i) + 'prune',
                                        accu=accu,
                                        target_accu=0.70, log=log)

    logclose()

if __name__ == "__main__":
    main()
