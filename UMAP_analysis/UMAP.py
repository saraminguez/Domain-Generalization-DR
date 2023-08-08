import torch
import pandas as pd
import os
import pytorch_lightning as pl
import argparse
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.models as models
import torchmetrics
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningModule
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from typing import Optional
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from torchmetrics import MaxMetric, MeanMetric
import torch.nn as nn
import umap
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm 
import shutil
from sklearn.utils.class_weight import compute_class_weight

import functions as f 

############### Variables #########################

# (Optional) command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--images_loc', nargs='+', type=str, default='C:/Users/Sara9/Documents/TFM/prueba', help='Data location')
parser.add_argument('--labels_loc', type=str, required=True, default='C:/Users/Sara9/Documents/TFM/Labels.csv', help='Labels location')
#parser.add_argument('--labels_umap_loc', type=str, required=True, default='C:/Users/Sara9/Documents/TFM/Labels.csv', help='Labels umap location')
#parser.add_argument('--images_test_loc', nargs='+', type=str, required=True, default='C:/Users/Sara9/Documents/TFM/prueba', help='Data location')
parser.add_argument('--lr', type=float, default=1e-2, help="Learning rate for the optimizer")
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--log_dir', required=True, type=str, help='Base log folder')
parser.add_argument('--max_epochs', type=int, default=10, help='Number of maximum epochs')
parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for dataloader')
parser.add_argument('--checkpoint_path', type=str, help='Checkpoint path')
parser.add_argument('--folder_embedding_name', type=str, help='Checkpoint path')

args = parser.parse_args()



############### Create dataframe ######################

labels_path = args.labels_loc

if len(args.images_loc) >1:
    print('Joint training')
    train_dfs = []
    val_dfs = []
    test_dfs = [] 

    for images_path in args.images_loc: 

        # load the dataframes
        df_train, df_val, df_test = f.get_dataframes(images_path, labels_path)
        
        train_dfs.append(df_train)
        val_dfs.append(df_val)
        test_dfs.append(df_test)
        
    df_train = pd.concat(train_dfs)
    df_test = pd.concat(test_dfs)
    df_val = pd.concat(val_dfs)

    df_train = df_train.sample(frac=1, random_state=random.randint(0, 100))
    df_val = df_val.sample(frac=1, random_state=random.randint(0, 100))
    df_test = df_test.sample(frac=1, random_state=random.randint(0, 100))

    # Apply the function to the 'camera_label' column
    df_train['labels_domain'] = f.map_camera_labels(df_train['labels_domain'])
    df_val['labels_domain'] = f.map_camera_labels(df_val['labels_domain'])
    df_test['labels_domain'] = f.map_camera_labels(df_test['labels_domain'])

    # Compute the class weights for the classifier
    class_weights_labels = compute_class_weight('balanced', classes =df_train['labels_class'].unique(), y = df_train['labels_class'])
    class_weights_labels = torch.from_numpy(class_weights_labels).float()
    class_weights_labels = class_weights_labels.to("cuda:0")
    print(class_weights_labels)

    # Compute the class weights for the discriminator
    class_weights_domain = compute_class_weight('balanced', classes =df_train['labels_domain'].unique(), y = df_train['labels_domain'])
    class_weights_domain = torch.from_numpy(class_weights_domain).float()
    class_weights_domain = class_weights_domain.to("cuda:0")
    print(class_weights_domain)


if len(args.images_loc) == 1:
    print('Single training')

    # load the dataframe
    df_train, df_val, df_test = f.get_dataframes(args.images_loc[0], labels_path)

df_train['labels_domain'] = f.map_camera_labels(df_train['labels_domain'])
df_val['labels_domain'] = f.map_camera_labels(df_val['labels_domain'])
df_test['labels_domain'] = f.map_camera_labels(df_test['labels_domain'])

# create the transform object
simple_transform = transforms.ToTensor()

# create the transform object
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.44740352034568787, 0.3267504870891571, 0.25300654768943787], std=[0.2815197706222534, 0.2094719558954239, 0.1712329387664795]),
    transforms.RandomRotation(degrees=(0, 360)),
    transforms.RandomResizedCrop(size=(512, 512), scale=(1.0, 1.35)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip()
])

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.44740352034568787, 0.3267504870891571, 0.25300654768943787], std = [0.2815197706222534, 0.2094719558954239, 0.1712329387664795])
])


############### Define model #######################

class RD_dataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image_path = self.df.iloc[idx, 0]
        image = f.imread(image_path)
        #get label class
        label_c = self.df.iloc[idx, 1]
        #get label domain 
        label_d = self.df.iloc[idx, 2]
        
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label_c, label_d, image_path
    
    
##########################################################################
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.feature_extractor = models.resnet50(weights ='DEFAULT')
        self.feature_extractor.fc = nn.Identity()
        
    def forward(self, x):
        x = self.feature_extractor(x)
        return x

##########################################################################
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier = torch.nn.Linear(2048, 5)

    def forward(self, x):
        x = self.classifier(x)
        return x

    
 ############### Define model #######################
     
class ResNetTransfer(LightningModule):
    
    def __init__(self, num_classes = 5, learning_rate: Optional[float] =  None, patience=2, momentum = 0.9, weight_decay = 0.0005, factor=0.1, verbose=True):
        super().__init__()
        #model
        
        self.feature_extractor = FeatureExtractor()

        self.classifier = Classifier()

        #learning rate
        self.learning_rate = learning_rate
        self.patience = patience
        self.factor = factor
        self.verbose = verbose


        self.momentum= momentum 
        self.nesterov = True
        self.weight_decay = weight_decay

        #Metrics for the classifier
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=5)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=5)
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=5)
        self.train_qwk = torchmetrics.classification.MulticlassCohenKappa(num_classes=5, weights = 'quadratic')
        self.val_qwk = torchmetrics.classification.MulticlassCohenKappa(num_classes=5, weights = 'quadratic')
        self.test_qwk = torchmetrics.classification.MulticlassCohenKappa(num_classes=5, weights = 'quadratic')

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.val_qwk_best = MaxMetric()

    def forward(self, x):
        # Extract the high-level feature representation
        features = self.feature_extractor(x)
        
        # Pass the features through the linear classifier
        logits_c = self.classifier(features)

        return features, logits_c


    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()
        self.val_qwk_best.reset()

    
    def model_step(self, batch, batch_idx):
        x, y_class, _ = batch

        features, logits_c = self(x, batch_idx=batch_idx)

        loss_c = F.cross_entropy(logits_c, y_class) 

        return loss_c, logits_c, y_class, features

    
    
    def training_step(self, batch, batch_idx: int):
        loss_c, logits_c, y_class, features = self.model_step(batch, batch_idx=batch_idx)

        # update and log metrics
        self.train_loss.update(loss_c)
        self.train_acc(logits_c, y_class)
        self.train_qwk(logits_c, y_class)


        #self.train_auroc(preds,targets)
        self.log("train_loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_qwk", self.train_qwk, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss_c, "preds": logits_c, "targets": y_class}


    def validation_step(self, batch, batch_idx: int):
        loss_c, logits_c, y_class, features = self.model_step(batch, batch_idx=batch_idx)

        # update and log metrics
        self.val_loss(loss_c)
        self.val_acc(logits_c, y_class)
        self.val_qwk(logits_c, y_class)


        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_qwk", self.val_qwk, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss_c, "preds": logits_c, "targets": y_class}

    def test_step(self, batch, batch_idx: int):
        loss_c, logits_c, y_class, features = self.model_step(batch, batch_idx=batch_idx)

        # update and log metrics
        self.test_loss(loss_c)
        self.test_acc(logits_c, y_class)
        self.test_qwk(logits_c, y_class)

        #self.test_auroc(preds, targets)
        self.log("test_loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_qwk", self.test_qwk, on_step=False, on_epoch=True, prog_bar=True)


    def configure_optimizers(self):
        #optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        #optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, nesterov=self.nesterov, weight_decay=self.weight_decay)
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        print(optimizer)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=2e-5, verbose = True)
        #scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=200)
        #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 50)
        print(scheduler)
        return {
        'optimizer': optimizer,
        'lr_scheduler': scheduler,
        'monitor': 'val_loss'  # Specify the metric to track (e.g., validation loss)
        }
    

 ##########################################################################################################################   



class RD_DataModule(LightningDataModule):
    def __init__(self, train_df: pd.DataFrame, val_df: pd.DataFrame, df_test: pd.DataFrame, batch_size: int):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = df_test
        self.batch_size = batch_size
        self.prepare_data_per_node = False
        self._log_hyperparams = True
    
    def train_dataloader(self):
        return DataLoader(dataset = RD_dataset(self.train_df, train_transforms), batch_size=self.batch_size, num_workers = args.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset = RD_dataset(self.val_df, val_transforms), batch_size=self.batch_size, num_workers = args.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(dataset = RD_dataset(self.test_df, val_transforms), batch_size=self.batch_size, num_workers = args.num_workers, shuffle=False)
    


#datamodule
dm = RD_DataModule(df_train, df_val, df_test, batch_size = args.batch_size)

#model
model = ResNetTransfer.load_from_checkpoint(args.checkpoint_path, num_classes=5)

# Set the model to evaluation mode
model.eval()

# Create an empty list to store the extracted features
all_features = []
all_labels_class = []
all_labels_domain = []
all_images_path = []

# Loop over the batches in the dataloader
for batch in tqdm(dm.test_dataloader()):
    # Extract the input images from the batch
    images, labels_class, labels_domain, image_path = batch

    # Pass the input data through the model to extract the features
    features, logits = model(images)

    all_labels_class.append(labels_class.numpy())

    all_labels_domain.append(labels_domain.numpy())

    all_images_path.append(image_path)

    # Append the extracted features to the list
    all_features.append(features.detach().cpu().numpy())

# Stack the extracted features into a single numpy array
all_features = np.vstack(all_features)

all_labels_class = np.concatenate(all_labels_class)

all_labels_domain = np.concatenate(all_labels_domain)

all_images_path = np.concatenate(all_images_path)

dst_folder_path = f.create_folder(args.folder_embedding_name)

np.save("all_features.npy", all_features)

all_features_path = '/data/0minguez/tfm-sara-minguez-retinopatia-diabetica/all_features.npy'

np.save("all_labels_class.npy", all_labels_class)

np.save("all_labels_domain.npy", all_labels_domain)

np.save("all_images_path.npy", all_images_path)

all_labels_class_path = '/data/0minguez/tfm-sara-minguez-retinopatia-diabetica/all_labels_class.npy'

all_labels_domain_path = '/data/0minguez/tfm-sara-minguez-retinopatia-diabetica/all_labels_domain.npy'

images_path = '/data/0minguez/tfm-sara-minguez-retinopatia-diabetica/all_images_path.npy'

# Print the shape of the extracted features
print(all_features.shape)
print(len(all_labels_domain))

reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(all_features)

np.save("embedding.npy", embedding)

embedding_path = '/data/0minguez/tfm-sara-minguez-retinopatia-diabetica/embedding.npy'

shutil.move(all_features_path, dst_folder_path)
shutil.move(all_labels_class_path, dst_folder_path)
shutil.move(all_labels_domain_path, dst_folder_path)
shutil.move(embedding_path, dst_folder_path)
shutil.move(images_path, dst_folder_path)
            

# `embedding` now contains the reduced-dimensional embeddings
print(embedding.shape)  # Should output (n_samples, 2)


