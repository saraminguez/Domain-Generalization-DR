# -*- coding: utf-8 -*-
"""
@author: Sara Mínguez 

Cropped function modified from: https://github.com/sveitser/kaggle_diabetic/blob/master/convert.py

get_mean_std function modified from: https://github.com/YijinHuang/pytorch-classification/blob/master/utils/func.py 

This file contains the functions used in the notebooks.
"""

import os
import imageio.v3 as iio
import skimage.color
import skimage.filters
import cv2
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms 
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def imread(path_img):
    #image = iio.imread(path_img)
    image = Image.open(path_img)
        
    return image


def imwrite(path_img, img):
    #path_img contains also the name of the image
    iio.imwrite(path_img, img)


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

def create_mask(img):
    gray_image = skimage.color.rgb2gray(img)

    # blur the image to denoise
    blurred_image = skimage.filters.gaussian(gray_image, sigma=10.0)
    
    # Define treshold
    t = 0.05

    # create a binary mask with the threshold 
    binary_mask = blurred_image > t
    
    return binary_mask

def mask_image(img,mask):
    img = img.copy()
    img[mask<=0,...]=0
    return img


def preprocess_image(img, resize_size):
    
    #create mask
    mask_img = create_mask(img)
    
    shape_img = img.shape
    height_total = shape_img[0]
    width_total= shape_img[1]

    # Get the non-zero elements of the mask
    non_zero_rows, non_zero_cols = np.nonzero(mask_img)

    # Find the minimum and maximum indices of the non-zero elements
    min_row, max_row = np.min(non_zero_rows), np.max(non_zero_rows)
    min_col, max_col = np.min(non_zero_cols), np.max(non_zero_cols)

    # Get the dimensions of the part that is not zero
    height_retina = max_row - min_row + 1
    width_retina = max_col - min_col + 1

    # Remove mask 
    img = mask_image(img, mask_img)

    if width_retina + 20 >height_retina: 
        # Crop the image using index slicing CROP VERTICAL
        cropped_img = img[:, min_col-10:max_col+11]

        if height_total > width_retina + 20:
            d = int(((width_retina + 20) -height_retina)/2) 
            cropped_img = cropped_img[min_row-d:max_row+1+d, :]

        else:
            #Calculate padding 
            vertical_padding = int((width_retina+20-height_retina)/2)
            #Pad image
            cropped_img = cv2.copyMakeBorder(cropped_img, vertical_padding, vertical_padding, 0, 0, cv2.BORDER_CONSTANT, value=0)
    else: 
        cropped_img = img[min_row-10:max_row+11, :]
        d = int((height_retina+20 -width_retina)/2)
        cropped_img = cropped_img[:, min_col-d:max_col+1+d]

    img = cv2.resize(cropped_img, (resize_size,resize_size))
    
    return img

def square_bbox(img):
    w, h = img.size
    left = max((w - h) // 2, 0)
    upper = 0
    right = min(w - (w - h) // 2, w)
    lower = h
    return (left, upper, right, lower)

def cropped_images(img, resize_size):

    blurred = img.filter(ImageFilter.BLUR)
    blurred = np.array(blurred)
    shape_img = blurred.shape
    height_total = shape_img[0]
    width_total= shape_img[1]


    if width_total > 1.2 * height_total:
        left_max = blurred[:, : width_total // 32, :].max(axis=(0, 1)).astype(int)
        right_max = blurred[:, - width_total // 32:, :].max(axis=(0, 1)).astype(int)
        max_bg = np.maximum(left_max, right_max)

        foreground = (blurred > max_bg + 10).astype(np.uint8)
        bbox = Image.fromarray(foreground).getbbox()

        if bbox is None:
            print('No bbox')
        else:
            left, upper, right, lower = bbox
            # if we selected less than 80% of the original
            # height, just crop the square
            if right - left < 0.8 * height_total or lower - upper < 0.8 * height_total:
                print('bbox too small')
                bbox = None
    else:
        bbox = None

    if bbox is None:
        bbox = square_bbox(img)

    cropped = img.crop(bbox)
    img = cropped.resize([resize_size, resize_size], Image.ANTIALIAS)
    
    return img


def save_preprocess_images(src_path, dst_folder_path, resize_size):
    
    create_folder(dst_folder_path)   
    
    image_names_src = os.listdir(src_path)
    
    image_names_dst = os.listdir(dst_folder_path)
    
    for name in image_names_src:
        
        if name not in image_names_dst: 
        
            path_image = src_path + '/' + name
            img = imread(path_image)
            
            try: 
                #preprocess image (remove mask and resize)
                prep_image = cropped_images(img, resize_size)
                
            except: 
                print(path_image)
                continue
            
            dst_path = dst_folder_path + '/' + name
            
            #save preprocess image in destination folder
            imwrite(dst_path, prep_image)

def get_labels(image_path, csv_path):
    
    #list of images
    image_names_src = os.listdir(image_path)

    #remove 'jpeg'
    image_list = [os.path.splitext(image)[0] for image in image_names_src]

    # load the csv file into a dataframe
    df = pd.read_csv(csv_path)
    df = df.set_index('image_name')
    
    
    #get labels for degree of the disease in the same order as images
    labels_class = list(map(lambda x: df.loc[x, 'level'], image_list))
    
    #get labels for camera domain in the same order as images
    labels_domain = list(map(lambda x: df.loc[x, 'camera'], image_list))
    
    return labels_class, labels_domain
             
def create_dataframe(image_path, csv_path):
    
    #get path of the images
    images_path = [os.path.join(image_path, file) for file in os.listdir(image_path)]
    
    #get their corresponding labels
    labels_class, labels_domain = get_labels(image_path, csv_path)
    
    #create a dataframe
    data = {'Images_path':images_path, 'labels_class':labels_class, 'labels_domain':labels_domain} 
    df = pd.DataFrame(data)
    
    return df

def get_mean_std(train_df, batch_size, num_workers, RD_dataset): 
    # create the transform object
    simple_transform = transforms.ToTensor()
    loader = DataLoader(dataset = RD_dataset(train_df, simple_transform),
        batch_size= batch_size,
        num_workers = num_workers,
        shuffle=False
    )

    num_samples = 0.
    channel_mean = torch.Tensor([0., 0., 0.])
    channel_std = torch.Tensor([0., 0., 0.])
    for samples in tqdm(loader):
        X, _, _ = samples
        channel_mean += X.mean((2, 3)).sum(0)
        num_samples += X.size(0)
    channel_mean /= num_samples

    for samples in tqdm(loader):
        X, _, _ = samples
        batch_samples = X.size(0)
        X = X.permute(0, 2, 3, 1).reshape(-1, 3)
        channel_std += ((X - channel_mean) ** 2).mean(0) * batch_samples
    channel_std = torch.sqrt(channel_std / num_samples)

    mean, std = channel_mean.tolist(), channel_std.tolist()
    print('mean: {}'.format(mean))
    print('std: {}'.format(std))
    return mean, std

def separate_dataframes(df_train_val: pd.DataFrame): 
    train_df, val_df = train_test_split(df_train_val, test_size=0.2, random_state=0, stratify=df_train_val['labels_domain'])
    return train_df, val_df



def get_class_weights(train_df:pd.DataFrame ):
    # Compute class weights using scikit-learn
    train_labels = train_df['labels'].values
    class_weights = compute_class_weight(class_weight = "balanced", classes= np.unique(train_labels), y= train_labels)

    return class_weights

def get_dataframes(images_path, labels_path): 

    train_path = os.path.join(images_path, "train")
    val_path = os.path.join(images_path, "val")
    test_path = os.path.join(images_path, "test")


    # load the dataframe
    df_train= create_dataframe(train_path, labels_path)
    df_val= create_dataframe(val_path, labels_path)
    df_test = create_dataframe(test_path, labels_path)

    return df_train, df_val, df_test


def map_camera_labels(labels):
    unique_labels = sorted(set(labels))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    return [label_map[label] for label in labels]



    
    