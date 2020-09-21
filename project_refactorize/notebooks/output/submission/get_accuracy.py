# library import
import sys
import pandas as pd

def acc_score():
    my_rst = pd.read_csv('/home/shared/sykim/lab_kaggle_practice1/project_refactorize/notebooks/output/submission/submission_ensemble.csv')
    test_rst = pd.read_csv('/home/shared/sykim/lab_kaggle_practice1/project_refactorize/notebooks/output/submission/test_label.csv')

    if len(my_rst)!=len(test_rst):
        print("wrong length")
        return None
    
    cnt = 0
    for idx in range(1, len(my_rst)):
        if my_rst.iloc[idx, 1] == test_rst.iloc[idx, 1]:
            cnt += 1
    
    return round((cnt/len(my_rst))*100,2)
