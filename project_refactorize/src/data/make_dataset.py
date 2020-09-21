# library import 
import pandas as pd
from pathlib import Path

def make_dataset():
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

    return df_train, df_test, df_class