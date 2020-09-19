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


def ensemble_total(**ensemble_kwargs):
    # get from ensemble_kwargs
    seresnext50_pred = ensemble_kwargs['seresnext50_pred']
    seresnext50_pred_tta = ensemble_kwargs['seresnext50_pred_tta']
    efficientnetb3_pred = ensemble_kwargs['efficientnetb3_pred']
    efficientnetb3_pred_tta = ensemble_kwargs['efficientnetb3_pred_tta']

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

    return submission_ensemble

