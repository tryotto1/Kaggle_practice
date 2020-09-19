# python library import 
import sys 
import time
import math
import random
import numpy as np
import pandas as pd
from pathlib import Path
import glob
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageOps
from tqdm import tqdm, tqdm_notebook

# torch import
import torch
from torch import nn, cuda
from torch.autograd import Variable 
import torch.nn.functional as F
import torchvision as vision
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, SGD, Optimizer
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, ReduceLROnPlateau

# model import
import pretrainedmodels
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# module import
sys.path.append('/home/shared/sykim/lab_kaggle_practice1/project_refactorize')
from src.data.train_dataset import TrainDataset
from src.data.test_dataset import TestDataset
from src.features.data_transforms import data_transforms
from src.features.data_transforms2 import data_transforms_2
from src.model.Loss.FocalLoss import FocalLoss
from src.model.optimizer.AdamW import Adam
from src.model.optimizer.CosineAnnealingWithRestartsLR import CosineAnnealingWithRestartsLR
from src.model.Training.train_one_epoch import train_one_epoch

# warning
import os
import warnings

warnings.filterwarnings(action='ignore')
pd.set_option('display.max_columns', 200)


# seed value fix
# seed value을 고정해야 hyper parameter 바꿀 때마다 결과를 비교할 수 있습니다.
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

SEED = 42
seed_everything(SEED)

# get image path 
TRAIN_IMAGE_PATH = Path('/home/shared/sykim/lab_kaggle_practice1/project_refactorize/data/processed/3rd-ml-month-car-image-cropping-dataset/train_crop/')
TEST_IMAGE_PATH = Path('/home/shared/sykim/lab_kaggle_practice1/project_refactorize/data/processed/3rd-ml-month-car-image-cropping-dataset/test_crop/')
DATA_PATH = '/home/shared/sykim/lab_kaggle_practice1/project_refactorize/data/processed/kakr-3rd-copys/'

# fetch data, using path
df_train = pd.read_csv('/home/shared/sykim/lab_kaggle_practice1/project_refactorize/data/processed/kakr-3rd-copy/train.csv')
df_test = pd.read_csv('/home/shared/sykim/lab_kaggle_practice1/project_refactorize/data/processed/kakr-3rd-copy/test.csv')
df_class = pd.read_csv('/home/shared/sykim/lab_kaggle_practice1/project_refactorize/data/processed/kakr-3rd-copy/class.csv')
df_train.head()

# preprocess data
df_train['class'] = df_train['class'] - 1
df_train = df_train[['img_file', 'class']]
df_test = df_test[['img_file']]

# seresnext model start
sys.path.append('/home/shared/sykim/lab_kaggle_practice1/project_refactorize')
from src.model.Training.model_seresnext50 import seresnext_total

seresnext_kwargs = dict(
                SEED=SEED,
                train_seresnext=True,
                df_train=df_train,
                df_test=df_test,
            )
seresnext50_pred, seresnext50_pred_tta = seresnext_total(**seresnext_kwargs)

# efficientNest model start
sys.path.append('/home/shared/sykim/lab_kaggle_practice1/project_refactorize')
from src.model.Training.model_efficientnet import efficientNet_total

efficientNet_kwargs = dict(
                SEED=SEED,
                train_efficientnet=True,
                df_train=df_train,
                df_test=df_test,
            )
efficientnetb3_pred, efficientnetb3_pred_tta = efficientNet_total(**efficientNet_kwargs)

# ensemble models start - submit result
sys.path.append('/home/shared/sykim/lab_kaggle_practice1/project_refactorize')
from src.model.Training.model_ensemble import ensemble_total

ensemble_kwargs = dict(
                seresnext50_pred = seresnext50_pred,
                seresnext50_pred_tta = seresnext50_pred_tta,
                efficientnetb3_pred = efficientnetb3_pred,
                efficientnetb3_pred_tta=efficientnetb3_pred_tta,
            )
submission_ensemble = ensemble_total(**ensemble_kwargs)



'''
the below code will be modulized into different folder
'''

# training code
def train_model(num_epochs=60, accumulation_step=4, mixup_loss=False, cv_checkpoint=False, fine_tune=False,
                weight_file_name='weight_best.pt', y_true=None, **train_kwargs):
    # choose scheduler
    if fine_tune:
        lr = 0.00001
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.000025)   
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1)
    else:    
        lr = 0.01
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.025)
        eta_min = 1e-6
        T_max = 5
        T_mult = 2
        restart_decay = 1.0
        scheduler = CosineAnnealingWithRestartsLR(optimizer,T_max=T_max, eta_min=eta_min, T_mult=T_mult, restart_decay=restart_decay)
        
        train_result = {}
    
    print(weight_file_name)
    train_result['weight_file_name'] = weight_file_name
    
    best_epoch = -1
    best_score = 0.
    lrs = []
    score = []
    
    for epoch in range(num_epochs):
        
        start_time = time.time()        
        train_loss = train_one_epoch(model, criterion, train_loader, optimizer, mixup_loss, accumulation_step)
        val_loss, val_score = validation(model, criterion, valid_loader, y_true)
        
        score.append(val_score)
    
        # model save (score or loss?)
        if cv_checkpoint:
            if val_score > best_score:
                best_score = val_score
                train_result['best_epoch'] = epoch + 1
                train_result['best_score'] = round(best_score, 5)
                torch.save(model.state_dict(), weight_file_name)
        else:
            if val_loss < best_loss:
                best_loss = val_loss
                train_result['best_epoch'] = epoch + 1
                train_result['best_loss'] = round(best_loss, 5)
                torch.save(model.state_dict(), weight_file_name)

        elapsed = time.time() - start_time
        
        lr = [_['lr'] for _ in optimizer.param_groups]
        print("Epoch {} - train_loss: {:.4f}  val_loss: {:.4f}  cv_score: {:.4f}  lr: {:.6f}  time: {:.0f}s".format(
                epoch+1, train_loss, val_loss, val_score, lr[0], elapsed))
        
        for param_group in optimizer.param_groups:
            lrs.append(param_group['lr'])
                # scheduler update
        if fine_tune:
            if cv_checkpoint:
                scheduler.step(val_score)
            else:
                scheduler.step(val_loss)
        else:
            scheduler.step()
     
    return train_result, lrs, score

# validation
def validation(model, criterion, valid_loader, y_true):    
    model.eval()
    valid_preds = np.zeros((len(valid_dataset), num_classes))
    val_loss = 0.
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(valid_loader):

            inputs, targets = inputs.cuda(), targets.cuda()
            
            outputs = model(inputs).detach()
            loss = criterion(outputs, targets)
            valid_preds[i * batch_size: (i+1) * batch_size] = outputs.cpu().numpy()
            
            val_loss += loss.item() / len(valid_loader)
            
        y_pred = np.argmax(valid_preds, axis=1)
        val_score = f1_score(y_true, y_pred, average='micro')  
        
    return val_loss, val_score


# model run
train_seresnext = True
if train_seresnext:
    k_folds = 4
    num_classes = 196

    skf = StratifiedKFold(n_splits=k_folds, random_state=SEED)
    start_fold = 1
    end_fold = k_folds
    result_arr = []

    for i, (train_index, valid_index) in enumerate(skf.split(df_train['img_file'], df_train['class'])):
        fold = i + 1
        train_df = df_train.iloc[train_index, :].reset_index()
        valid_df = df_train.iloc[valid_index, :].reset_index()
        y_true = valid_df['class'].values

        print("===========================================")
        print("====== K Fold Validation step => %d/%d ======" % ((fold),k_folds))
        print("===========================================")

        batch_size = 16 * torch.cuda.device_count()

        train_dataset = TrainDataset(train_df, mode='train', transforms=data_transforms)
        valid_dataset = TrainDataset(valid_df, mode='valid', transforms=data_transforms)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        if fold >= start_fold and fold <= end_fold:
            torch.cuda.empty_cache()

            model = pretrainedmodels.se_resnext50_32x4d(pretrained='imagenet')
            model.last_linear = nn.Linear(2048, num_classes)
            
            model.cuda()

            criterion = FocalLoss()

            train_kwargs = dict(
                train_loader=train_loader,
                valid_loader=valid_loader,
                model=model,
                criterion=criterion,
            )

            num_epochs = 20
            result, lrs, score = train_model(num_epochs=num_epochs, accumulation_step=16, mixup_loss=False,
                                            cv_checkpoint=True, fine_tune=False,
                                            weight_file_name=f'seresnext50_fold_{fold}.pt',
                                            y_true=y_true, **train_kwargs)
            result_arr.append(result)
            print(result)


# test set prediction - seresNet
k_folds = 4
num_classes = 196
batch_size = 1
test_dataset = TestDataset(df_test, mode='test', transforms=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
total_num_models = k_folds

model = pretrainedmodels.se_resnext50_32x4d(pretrained=None)
model.last_linear = nn.Linear(2048, num_classes)
model.cuda()

all_prediction = np.zeros((len(test_dataset), num_classes))

for f in range(k_folds):
    fold = f + 1
    print(f'fold {fold} prediction starts')
    
    weight_path = f'/home/shared/sykim/lab_kaggle_practice1/project_refactorize/seresnext50_fold_{fold}.pt'    
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    prediction = np.zeros((len(test_dataset), num_classes)) # num_classes=196
    with torch.no_grad():
        for i, images in enumerate(test_loader):
            images = images.cuda()

            preds = model(images).detach()
            preds = F.softmax(preds, dim=1) # convert output to probability
            prediction[i * batch_size: (i+1) * batch_size] = preds.cpu().numpy()
        
        print(prediction)

    all_prediction = all_prediction + prediction

all_prediction /= total_num_models


seresnext50_pred = pd.DataFrame(all_prediction)
seresnext50_pred.to_csv('seresnext50_pred.csv', index=False)

seresnext50_pred.head()



# train - tta
k_folds = 4
num_classes = 196

batch_size = 1
tta = 3
tta_dataset = TestDataset(df_test, mode='tta', transforms=data_transforms)
tta_loader = DataLoader(tta_dataset, batch_size=batch_size, shuffle=False)
total_num_models = k_folds*tta

model = pretrainedmodels.se_resnext50_32x4d(pretrained=None)
model.last_linear = nn.Linear(2048, num_classes)
model.cuda()

all_prediction_tta = np.zeros((len(tta_dataset), num_classes))

for f in range(k_folds):
    fold = f + 1
    print(f'fold {fold} prediction starts')
    
    for _ in range(tta):
        print("tta {}".format(_+1))

        weight_path = f'/home/shared/sykim/lab_kaggle_practice1/project_refactorize/seresnext50_fold_{fold}.pt'
        model.load_state_dict(torch.load(weight_path))

        model.eval()
        
        prediction = np.zeros((len(tta_dataset), num_classes)) # num_classes=196
        with torch.no_grad():
            for i, images in enumerate(tta_loader):
                images = images.cuda()

                preds = model(images).detach()
                preds = F.softmax(preds, dim=1) # convert output to probability
                prediction[i * batch_size: (i+1) * batch_size] = preds.cpu().numpy()
        all_prediction_tta = all_prediction_tta + prediction
    
all_prediction_tta /= total_num_models

seresnext50_pred_tta = pd.DataFrame(all_prediction_tta)
seresnext50_pred_tta.to_csv('seresnext50_pred_tta.csv', index=False)

seresnext50_pred_tta.head()



# ensemble = seresNext + efficientNet
seresnext50_ensemble = 0.25*seresnext50_pred.values + 0.75*seresnext50_pred_tta.values
efficientnetb3_ensemble = 0.25*efficientnetb3_pred.values + 0.75*efficientnetb3_pred_tta.values
final_ensemble = 0.3*efficientnetb3_ensemble + 0.7*seresnext50_ensemble

result_ensemble = np.argmax(final_ensemble, axis=1)
result_ensemble = result_ensemble + 1

submission_ensemble = pd.read_csv('/home/shared/sykim/lab_kaggle_practice1/project_refactorize/data/processed/kakr-3rd-copy/sample_submission.csv')
submission_ensemble["class"] = result_ensemble
submission_ensemble.to_csv("submission_ensemble.csv", index=False)
submission_ensemble.head()

