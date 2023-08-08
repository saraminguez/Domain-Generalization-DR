# -*- coding: utf-8 -*-
"""
This file allow to separate the images from EyePACS dataset by their corresponding camera labels

@author: Sara Mínguez
"""

# Imports 
import pandas as pd 
import os
import shutil
import functions 

def get_quality_labels():
    quality_labels_train = pd.read_csv('D:/TFM/Datasets/EyePACS/Quality_labels/Label_EyeQ_train.csv')
    quality_labels_test = pd.read_csv('D:/TFM/Datasets/EyePACS/Quality_labels/Label_EyeQ_test.csv')
    
    quality_labels_train_names = quality_labels_train[quality_labels_train['quality'] == 2]['image'].tolist()
    quality_labels_test_names = quality_labels_test[quality_labels_test['quality'] == 2]['image'].tolist()
    quality_labels_train_names.extend(quality_labels_test_names)
    quality_labels_total = quality_labels_train_names
    
    return quality_labels_total

def create_files(quality_labels_total, kind):
    dataset = pd.read_csv("C:/Users/Sara9/Documents/TFM/pytorch-classification-master/eyepace_all_manufacturer.csv")
    labels = dataset['camera'].unique()
    src_folder = ("C:/Users/Sara9/Documents/TFM/pytorch-classification-master")
    test_folder = ("C:/Users/Sara9/Documents/TFM/pytorch-classification-master/test_new_preprocessed")
    train_folder = ("C:/Users/Sara9/Documents/TFM/pytorch-classification-master/train_new_preprocessed")
    folder_test_name = 'label_list_'+ str(1) + '_test'
    folder_test_1 = os.path.join(src_folder, folder_test_name)
    functions.create_folder(folder_test_1)
       
    
    if kind == 'test': 
        for label in labels:
            folder_name = 'label_list_'+ str(label)
            folder_direction = os.path.join(src_folder, folder_name)
            functions.create_folder(folder_direction)
            image_labels = dataset[dataset['camera'] == str(label)]['image_name'].tolist()
            names = os.listdir("C:/Users/Sara9/Documents/TFM/pytorch-classification-master/test_new_preprocessed")
            for image in image_labels:
                if image not in quality_labels_total:
                    if image in names:
                        if label == str(1):
                            src_path = os.path.join(test_folder, image)
                            dst_path = os.path.join(folder_test_1, image)
                            shutil.move(src_path, dst_path)
                        else:
                            src_path = os.path.join(test_folder, image)
                            dst_path = os.path.join(folder_direction, image)
                            shutil.move(src_path, dst_path)
                                                               
                        
    else:
            for label in labels: 
                image_labels = dataset[dataset['camera'] == str(label)]['image_name'].tolist()
                names = os.listdir("C:/Users/Sara9/Documents/TFM/pytorch-classification-master/train_new_preprocessed")
                folder_name = 'label_list_'+ str(label)
                folder_direction = os.path.join(src_folder, folder_name)
                functions.create_folder(folder_direction)
                for image in image_labels:
                    if image not in quality_labels_total:
                        if image in names:
                            src_path = os.path.join(train_folder, image)
                            dst_path = os.path.join(folder_direction, image)
                            shutil.move(src_path, dst_path)