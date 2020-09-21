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
from src.data.make_dataset import make_dataset
from src.model.Training.model_seresnext50 import seresnext_total
from src.model.Training.model_efficientnet import efficientNet_total
from src.model.Training.model_ensemble import ensemble_total
from notebooks.output.submission.get_accuracy import acc_score
from notebooks.fix_seed import seed_everything

# warning
import os
import warnings

# main function
if __name__ == "__main__":
    # set warning sign
    warnings.filterwarnings(action='ignore')
    pd.set_option('display.max_columns', 200)

    # fix seed value
    SEED = 42
    seed_everything(SEED)

    # make dataset
    df_train, df_test, df_class = make_dataset()

    # seresnext model start
    seresnext_kwargs = dict(
                    SEED=SEED,
                    train_seresnext=True,
                    df_train=df_train,
                    df_test=df_test,
                )
    seresnext50_pred, seresnext50_pred_tta = seresnext_total(**seresnext_kwargs)

    # efficientNest model start
    efficientNet_kwargs = dict(
                    SEED=SEED,
                    train_efficientnet=True,
                    df_train=df_train,
                    df_test=df_test,
                )
    efficientnetb3_pred, efficientnetb3_pred_tta = efficientNet_total(**efficientNet_kwargs)

    # ensemble models start - submit result
    seresnext50_pred = pd.read_csv('/home/shared/sykim/lab_kaggle_practice1/project_refactorize/notebooks/seresnext50_pred.csv')
    seresnext50_pred_tta = pd.read_csv('/home/shared/sykim/lab_kaggle_practice1/project_refactorize/notebooks/seresnext50_pred_tta.csv')
    efficientnetb3_pred = pd.read_csv('/home/shared/sykim/lab_kaggle_practice1/project_refactorize/notebooks/efficientnetb3_pred.csv')
    efficientnetb3_pred_tta = pd.read_csv('/home/shared/sykim/lab_kaggle_practice1/project_refactorize/notebooks/efficientnetb3_pred_tta.csv')

    ensemble_kwargs = dict(
                    seresnext50_pred = seresnext50_pred,
                    seresnext50_pred_tta = seresnext50_pred_tta,
                    efficientnetb3_pred = efficientnetb3_pred,
                    efficientnetb3_pred_tta=efficientnetb3_pred_tta,
                )
    submission_ensemble = ensemble_total(**ensemble_kwargs)

    # calculate accuracy
    rst_acc = acc_score()
    print("final accuracy : " + str(rst_acc) + "%")
