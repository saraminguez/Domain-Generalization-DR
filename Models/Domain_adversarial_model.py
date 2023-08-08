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
import matplotlib.pyplot as plt
import seaborn as sns
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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau


############### Variables #########################

# (Optional) command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--images_loc', nargs='+', type=str, default='C:/Users/Sara9/Documents/TFM/prueba', help='Data location')
parser.add_argument('--labels_loc', type=str, required=True, default='C:/Users/Sara9/Documents/TFM/Labels.csv', help='Labels location')
parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for the optimizer")
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--log_dir', required=True, type=str, help='Base log folder')
parser.add_argument('--max_epochs', type=int, default=10, help='Number of maximum epochs')
parser.add_argument('--num_workers', type=int, default=12, help='Number of workers for dataloader')
parser.add_argument('--final_checkpoint_name', type=str, help='Final checkpoint name')
parser.add_argument('--color_transformations', type=str, help='Perform color transformations or not: color and augmix')
parser.add_argument('--mode', type=str, required=True, help='Training or evaluation')
parser.add_argument('--checkpoint_path', type=str, help='Checkpoint path')
parser.add_argument('--es_patience', type=int, help='Early stopping patience')
parser.add_argument('--lambda_value', type=float, default=0.1, help='Lambda value for the adversarial loss')
parser.add_argument('--seed', type=int, help='Seed for reproducibility')

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
"""
class Discriminator(nn.Module):
    def __init__(self, num_domains = 3, feature_dim=2048, hidden_size=1024):
        super(Discriminator, self).__init__()

        self.discriminator = nn.Sequential(
            nn.Linear(in_features=feature_dim, out_features=hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, out_features=num_domains))

    def forward(self, x):
        x = self.discriminator(x)
        return x   
"""
        
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


    def forward(self, x, batch_idx):
        # Extract the high-level feature representation
        features = self.feature_extractor(x)
        
        # Pass the features through the linear classifier
        logits_c = self.classifier(features)

        if self.training:
          max_batches = len(dm.train_dataloader())
          p = float(batch_idx + self.current_epoch * max_batches) / (self.trainer.max_epochs * max_batches)
          #self.lr_schedule_step(p)
          lambda_p = self.get_lambda_p(p=p)
          #lambda_p = args.lambda_value 
          features_grl = GRL.apply(features, lambda_p)
          logits_d = self.discriminator(features_grl)
          return features, logits_c, logits_d

        else:
          logits_d = self.discriminator(features)
          return features, logits_c, logits_d


    def training_step(self, batch, batch_idx):

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
        loss_c = F.cross_entropy(logits_c, y_class, weight=class_weights_labels)
        # for averaging loss across batches
        self.train_loss_c(loss_c)
        self.log("train_loss_c", self.train_loss_c, on_step=False, on_epoch=True, prog_bar=True)

        #Compute the loss for the discriminator
        loss_d = F.cross_entropy(logits_d, y_domain, weight=class_weights_domain)
        self.train_loss_d(loss_d)
        self.log("train_loss_d", self.train_loss_d, on_step=False, on_epoch=True, prog_bar=True)

        #Compute the total loss
        loss = loss_c + args.lambda_value * loss_d
        self.train_loss(loss)
        self.log("train_loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):

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

        loss = loss_c +  args.lambda_value * loss_d
        self.val_loss(loss)
        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):

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

        loss = loss_c + args.lambda_value * loss_d
        self.test_loss(loss)
        self.log("test_loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def configure_optimizers(self):
      optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
      scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=5, verbose = True)
      #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 50)
      print(scheduler)
      return {
        'optimizer': optimizer,
        'lr_scheduler': scheduler,
        'monitor': 'val_qwk'  # Specify the metric to track (e.g., validation loss)
      }

                

    def get_lambda_p(self, p):
        gamma = 10
        lambda_p = 2. / (1. + np.exp(-gamma * p)) - 1

        return lambda_p
        
#################################################################################################################################

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
        
        return image, label_c, label_d 


#Predifined values for mean and std for ImageNet
#mean = [0.485, 0.456, 0.406]
#std = [0.229, 0.224, 0.225]
mean, std = f.get_mean_std(df_train, args.batch_size, args.num_workers, RD_dataset)

geometric_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
    transforms.RandomRotation(degrees=(0, 360)),
    transforms.RandomApply([transforms.RandomResizedCrop(size=(512, 512), scale=(1.0, 1.35), antialias=True)], p=0.5),
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
    transforms.Normalize(mean=mean, std=std)])

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

if args.color_transformations == 'color':
  train_transforms = color_transforms
  print(train_transforms)

if args.color_transformations == 'augmix':
  
  train_transforms = augmix_transforms
  print(train_transforms)

if args.color_transformations == 'no':
    train_transforms = geometric_transforms
    print(train_transforms)
    
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
    

#################################################################################################################################

experiments_dir = args.log_dir

if not os.path.exists(experiments_dir):
    os.makedirs(experiments_dir)

logger = TensorBoardLogger(save_dir = experiments_dir)

#datamodule
dm = RD_DataModule(df_train, df_val, df_test, batch_size = args.batch_size)

model = LitModuleClasAdversarial(batch_size=args.batch_size, learning_rate = args.lr)


early_stop_callback = EarlyStopping(
   monitor='val_qwk',
   patience=args.es_patience,
   verbose=True,
   mode='max'
)


final_checkpoint_callback = ModelCheckpoint(
    save_last=True,
    dirpath="final_adversarial_checkpoints/",
    filename="final-" + str(args.final_checkpoint_name)
)

best_checkpoint_callback  = ModelCheckpoint(
    dirpath='final_adversarial_checkpoints/', 
    filename='best-{epoch:02d}-{val_qwk:.5f}',
    save_last = True, 
    save_top_k=1,
    monitor='val_qwk', 
    mode='max', 
    verbose = True
)

lr_monitor = LearningRateMonitor(logging_interval='epoch') 

##################### Training ###########################   
if args.mode == 'train':
    print('Training mode')    
    # Trainer(accelerator='gpu', devices=1)

    seed_everything(args.seed, workers=True)
  
    trainer = pl.Trainer(
        deterministic=True,
        default_root_dir = experiments_dir,
        max_epochs = args.max_epochs, 
        accelerator='gpu',
        devices=1,
        callbacks=[best_checkpoint_callback, final_checkpoint_callback, early_stop_callback,
                lr_monitor], 
        logger = logger)  

    # early_stop_callback,
        
    trainer.fit(model, dm)  

#################### Testing ################################

    result = trainer.test(model, dm)

##################### Evaluation ###########################   

if args.mode == 'eval':
    print('Evaluation mode') 
    model = LitModuleClasAdversarial.load_from_checkpoint(args.checkpoint_path, batch_size = args.batch_size)
    trainer = pl.Trainer(accelerator='gpu',devices=1)

    trainer.test(model, dm, ckpt_path=args.checkpoint_path) 
