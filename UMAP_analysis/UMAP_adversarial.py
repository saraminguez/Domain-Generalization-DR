"""
@author: Sara Mínguez

""" 
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import random
import functions as f
import argparse
import torchvision.models as models
import torch.nn as nn
from typing import Optional
import os
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics
from pytorch_lightning import LightningDataModule
import pandas as pd
import torch.optim as optim
import torchvision.transforms as transforms
from pytorch_lightning.utilities.data import CombinedLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import LightningModule
from sklearn.utils.class_weight import compute_class_weight
from torchmetrics import MaxMetric, MeanMetric
from pytorch_lightning import seed_everything
import umap
from tqdm import tqdm 
import shutil

# (Optional) command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--images_loc', nargs='+', type=str, default='C:/Users/Sara9/Documents/TFM/prueba', help='Data location')
parser.add_argument('--labels_loc', type=str, required=True, default='C:/Users/Sara9/Documents/TFM/Labels.csv', help='Labels location')
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

##########################################################################
class GRL(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, alpha):
    ctx.alpha = alpha

    return x.view_as(x)

  @staticmethod
  def backward(ctx, grad_output):
    output = grad_output.neg() * ctx.alpha

    return output, None

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
    def __init__(self, num_classes = 5):
        super(Classifier, self).__init__()
        self.classifier = torch.nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.classifier(x)
        return x

##########################################################################

class Discriminator(nn.Module):
    def __init__(self, num_domains = 3, feature_dim=2048):
        super(Discriminator, self).__init__()
        
        #num_domains = 3 # 3 domains: source, target, target2 as in joint training
        self.discriminator =nn.Sequential(
        nn.Linear(in_features=feature_dim, out_features=1024),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(in_features=1024, out_features=1024),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(in_features=1024, out_features=num_domains))

    def forward(self, x):
        x = self.discriminator(x)
        return x 

##########################################################################
   

class LitModuleClasAdversarial(LightningModule):

    """A LightningModule organizes your PyTorch code into 6 sections:

    - Computations (init)

    - Train loop (training_step)

    - Validation loop (validation_step)

    - Test loop (test_step)

    - Prediction Loop (predict_step)

    - Optimizers and LR Schedulers (configure_optimizers)

    Docs:

    https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html"""

    def __init__(self,batch_size, learning_rate: Optional[float] =  None, patience=2, num_classes=5, momentum = 0.9, weight_decay = 0.0005, factor=0.1, verbose=True, num_domains=2):
        super().__init__()

        self.feature_extractor = FeatureExtractor()

        self.classifier = Classifier()

        self.discriminator = Discriminator()


        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.factor = factor
        self.verbose = verbose
        self.momentum= momentum 
        self.nesterov = True
        self.weight_decay = weight_decay

        #Metrics for the classifier
        self.class_train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=5)
        self.class_val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=5)
        self.class_test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=5)
        self.train_qwk = torchmetrics.classification.MulticlassCohenKappa(num_classes=5, weights = 'quadratic')
        self.val_qwk = torchmetrics.classification.MulticlassCohenKappa(num_classes=5, weights = 'quadratic')
        self.test_qwk = torchmetrics.classification.MulticlassCohenKappa(num_classes=5, weights = 'quadratic')

        #Metrics for the discriminator
        self.domain_train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=3)
        self.domain_val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=3)
        self.domain_test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=3)

        # for averaging loss across batches
        self.train_loss_c = MeanMetric()
        self.val_loss_c = MeanMetric()
        self.test_loss_c = MeanMetric()

        self.train_loss_d = MeanMetric()
        self.val_loss_d = MeanMetric()
        self.test_loss_d = MeanMetric()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()


    def forward(self, x, batch_idx=None):
        # Extract the high-level feature representation
        features = self.feature_extractor(x)
        
        # Pass the features through the linear classifier
        logits_c = self.classifier(features)

        if self.training:
          max_batches = len(dm.train_dataloader())
          p = float(batch_idx + self.current_epoch * max_batches) / (self.trainer.max_epochs * max_batches)
          lambda_p = self.get_lambda_p(p=p)
          features_grl = GRL.apply(features, lambda_p)
          logits_d = self.discriminator(features_grl)
          return features, logits_c, logits_d

        else:
          logits_d = self.discriminator(features)
          return features, logits_c, logits_d


    def training_step(self, batch, batch_idx:int):

        x, y_class, y_domain = batch

        features, logits_c, logits_d = self(x, batch_idx=batch_idx)

        #Metrics for the classifier
        self.class_train_acc(logits_c, y_class)
        self.train_qwk(logits_c, y_class)
        self.log("train_qwk", self.train_qwk, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("class_train_acc", self.class_train_acc, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        #Metrics for the discriminator
        self.domain_train_acc(logits_d, y_domain)
        self.log("domain_train_acc", self.domain_train_acc, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        #Compute the loss for the classifier
        loss_c = F.cross_entropy(logits_c, y_class)
        # for averaging loss across batches
        self.train_loss_c(loss_c)
        self.log("train_loss_c", self.train_loss_c, on_step=False, on_epoch=True, prog_bar=True)

        #Compute the loss for the discriminator
        loss_d = F.cross_entropy(logits_d, y_domain, weight=class_weights_domain)
        self.train_loss_d(loss_d)
        self.log("train_loss_d", self.train_loss_d, on_step=False, on_epoch=True, prog_bar=True)

        #Compute the total loss
        loss = loss_c + loss_d
        self.train_loss(loss)
        self.log("train_loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx:int):

        x, y_class, y_domain = batch

        features, logits_c, logits_d = self(x, batch_idx=batch_idx)


        #Metrics for the classifier
        self.class_val_acc(logits_c, y_class)
        self.val_qwk(logits_c, y_class)
        self.log("val_qwk", self.val_qwk, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("class_val_acc", self.class_val_acc, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        #Metrics for the discriminator
        self.domain_val_acc(logits_d, y_domain)
        self.log("domain_val_acc", self.domain_val_acc, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

         #Compute the loss for the classifier
        loss_c = F.cross_entropy(logits_c, y_class)
        # for averaging loss across batches
        self.val_loss_c(loss_c)
        self.log("val_loss_c", self.val_loss_c, on_step=False, on_epoch=True, prog_bar=True)

        loss_d = F.cross_entropy(logits_d, y_domain)
        self.val_loss_d(loss_d)
        self.log("val_loss_d", self.val_loss_d, on_step=False, on_epoch=True, prog_bar=True)

        loss = loss_c + loss_d
        self.val_loss(loss)
        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx:int):

        x, y_class, y_domain = batch

        features, logits_c, logits_d = self(x, batch_idx=batch_idx)

        #Metrics for the classifier
        self.class_test_acc(logits_c, y_class)
        self.test_qwk(logits_c, y_class)
        self.log("test_qwk", self.test_qwk, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("class_test_acc", self.class_test_acc, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        #Metrics for the discriminator
        self.domain_test_acc(logits_d, y_domain)
        self.log("domain_test_acc", self.domain_test_acc, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        loss_c = F.cross_entropy(logits_c, y_class)
        # for averaging loss across batches
        self.test_loss_c(loss_c)
        self.log("test_loss_c", self.test_loss_c, on_step=False, on_epoch=True, prog_bar=True)

        loss_d = F.cross_entropy(logits_d, y_domain)
        self.test_loss_d(loss_d)
        self.log("test_loss_d", self.test_loss_d, on_step=False, on_epoch=True, prog_bar=True)

        loss = loss_c + loss_d
        self.test_loss(loss)
        self.log("test_loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def configure_optimizers(self):
      optimizer = optim.SGD([
                  {'params': self.feature_extractor.parameters()},
                  {'params': self.classifier.parameters(), 'lr': 1e-2}, 
                  {'params': self.discriminator.parameters(), 'lr': 1e-4}
              ], lr=1e-3, momentum=0.9,weight_decay = 0.0005)
      scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 25)

      return [optimizer], [scheduler]
                

    def get_lambda_p(self, p):
        gamma = 10
        lambda_p = 2. / (1. + np.exp(-gamma * p)) - 1

        return lambda_p
    
  ########################################################################################
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

    
#mean, std = f.get_mean_std(df_train_src, args.batch_size, args.num_workers, RD_dataset)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

geometric_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
    transforms.RandomRotation(degrees=(0, 360)),
    transforms.RandomApply([transforms.RandomResizedCrop(size=(512, 512), scale=(1.0, 1.35),  antialias=True)], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip()
])

color_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
    transforms.RandomRotation(degrees=(0, 360)),
    transforms.RandomApply([transforms.RandomResizedCrop(size=(512, 512), scale=(1.0, 1.35),  antialias=True)], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(), 
    transforms.RandomApply([transforms.ColorJitter(brightness=0.3,
            contrast=0.3,
            saturation=0.3
        )],p=0.5)
])

augmix_transforms = transforms.Compose([
    transforms.AugMix(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
    transforms.RandomRotation(degrees=(0, 360)),
    transforms.RandomApply([transforms.RandomResizedCrop(size=(512, 512), scale=(1.0, 1.35),  antialias=True)], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip()])

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

train_transforms = geometric_transforms
#print(train_transforms)
    
#################################################################################################################################
class RD_DataModule(LightningDataModule):
    def __init__(self, train_df: pd.DataFrame, val_df: pd.DataFrame, df_test: pd.DataFrame, batch_size: int):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = df_test
        self.batch_size = batch_size
        self.prepare_data_per_node = False
        self._log_hyperparams = True
    
    def train_dataloader(self):
        return DataLoader(dataset = RD_dataset(self.train_df, train_transforms), batch_size=self.batch_size, num_workers = args.num_workers, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(dataset = RD_dataset(self.val_df, val_transforms), batch_size=self.batch_size, num_workers = args.num_workers, shuffle=False, drop_last=True)

    def test_dataloader(self):
        return DataLoader(dataset = RD_dataset(self.test_df, val_transforms), batch_size=self.batch_size, num_workers = args.num_workers, shuffle=False, drop_last=True)
    

##########################################################################################
experiments_dir = args.log_dir

if not os.path.exists(experiments_dir):
    os.makedirs(experiments_dir)

logger = TensorBoardLogger(save_dir = experiments_dir)

dm = RD_DataModule(df_train, df_val, df_test, batch_size = args.batch_size)

#model
model = LitModuleClasAdversarial.load_from_checkpoint(args.checkpoint_path, num_classes=5, batch_size = args.batch_size)

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
    features, logits_c, logits_d = model(images)

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