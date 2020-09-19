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

# train - efficientnet-b3
def efficientNet_total(**efficientNet_kwargs):
    # get from efficientNet_kwargs
    SEED = efficientNet_kwargs['SEED']
    train_efficientnet = efficientNet_kwargs['train_efficientnet']
    df_train = efficientNet_kwargs['df_train']
    df_test = efficientNet_kwargs['df_test']

    # training start
    model_name = 'efficientnet-b3'
    image_size = EfficientNet.get_image_size(model_name)
    print(image_size)
    
    if train_efficientnet:
        k_folds = 4
        num_classes = 196
        skf = StratifiedKFold(n_splits=k_folds, random_state=SEED)
        start_fold = 1
        end_fold = 1
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

            train_dataset = TrainDataset(train_df, mode='train', transforms=data_transforms_2)
            valid_dataset = TrainDataset(valid_df, mode='valid', transforms=data_transforms_2)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

            if fold >= start_fold and fold <= end_fold:
                torch.cuda.empty_cache()

                model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
                
                if torch.cuda.device_count() > 1:
                    print(f'use multi gpu : {torch.cuda.device_count()}')
                    model = nn.DataParallel(model)
                model.cuda()

                criterion = nn.CrossEntropyLoss()

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

                num_epochs = 75
                result, lrs, score = train_model(num_epochs=num_epochs, accumulation_step=16, mixup_loss=False,
                                                cv_checkpoint=True, fine_tune=False, weight_file_name=f'efficientnetb3_fold_{fold}.pt',
                                                y_true=y_true, **train_kwargs)
                result_arr.append(result)
                print(result)

    # test - efficientNet
    k_folds = 4
    num_classes = 196

    batch_size = 1
    test_dataset = TestDataset(df_test, mode='test', transforms=data_transforms_2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    total_num_models = k_folds

    model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
    model.cuda()

    all_prediction = np.zeros((len(test_dataset), num_classes))

    for f in range(k_folds):
        fold = f + 1
        print(f'fold {fold} prediction starts')
        
        weight_path = f'/home/shared/sykim/lab_kaggle_practice1/project_refactorize/efficientnetb3_fold_{fold}.pt'
        model.load_state_dict(torch.load(weight_path))

        
        model.eval()

        prediction = np.zeros((len(test_dataset), num_classes)) # num_classes=196
        with torch.no_grad():
            for i, images in enumerate(test_loader):
                images = images.cuda()

                preds = model(images).detach()
                preds = F.softmax(preds, dim=1) # convert output to probability
                prediction[i * batch_size: (i+1) * batch_size] = preds.cpu().numpy()
        all_prediction = all_prediction + prediction
        
    all_prediction /= total_num_models



    seresnext50_pred = pd.DataFrame(all_prediction)
    seresnext50_pred.to_csv('seresnext50_pred.csv', index=False)

    seresnext50_pred.head()


    efficientnetb3_pred = pd.DataFrame(all_prediction)
    efficientnetb3_pred.to_csv('efficientnetb3_pred.csv', index=False)

    efficientnetb3_pred.head()


    # train - efficient net tta
    k_folds = 4
    num_classes = 196

    batch_size = 1
    tta = 3
    tta_dataset = TestDataset(df_test, mode='tta', transforms=data_transforms_2)
    tta_loader = DataLoader(tta_dataset, batch_size=batch_size, shuffle=False)
    total_num_models = k_folds*tta

    model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
    model.cuda()

    all_prediction_tta = np.zeros((len(tta_dataset), num_classes))

    for f in range(k_folds):
        fold = f + 1
        print(f'fold {fold} prediction starts')
        
        for _ in range(tta):
            print("tta {}".format(_+1))
            
            weight_path = f'/home/shared/sykim/lab_kaggle_practice1/project_refactorize/efficientnetb3_fold_{fold}.pt'
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

    efficientnetb3_pred_tta = pd.DataFrame(all_prediction_tta)
    efficientnetb3_pred_tta.to_csv('efficientnetb3_pred_tta.csv', index=False)

    efficientnetb3_pred_tta.head()

    return efficientnetb3_pred, efficientnetb3_pred_tta
