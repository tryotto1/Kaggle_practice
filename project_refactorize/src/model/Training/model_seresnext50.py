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
from src.model.optimizer.AdamW import AdamW
from src.model.optimizer.CosineAnnealingWithRestartsLR import CosineAnnealingWithRestartsLR
from src.model.Training.train_one_epoch import train_one_epoch
from src.model.Training.train_total import train_model

def seresnext_total(**seresnext_kwargs):
    # get from seresnext_kwargs
    SEED = seresnext_kwargs['SEED']
    train_seresnext = seresnext_kwargs['train_seresnext']
    df_train = seresnext_kwargs['df_train']
    df_test = seresnext_kwargs['df_test']
    
    # training start
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

                    # for validation
                    valid_df=valid_df,
                    num_classes=num_classes,
                    batch_size= batch_size,
                    valid_dataset=valid_dataset,
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

    return seresnext50_pred, seresnext50_pred_tta
