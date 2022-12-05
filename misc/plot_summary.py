import pandas as pd
import glob
import os
from misc import get_metrics
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
import csv

root = 'test crank-ms'
data = 'test crank-ms'
mode = 'svm'

dir_ = './{}/{}/{}'.format(root, data, mode)


roc_all, pr_all = [], []

csv_filepath = './{}/results'.format(dir_)
for csv_filename in os.listdir(csv_filepath):
    df = pd.read_csv('{}/{}'.format(csv_filepath, csv_filename))
    y_true = df.iloc[:,0]
    scores = df.iloc[:,2]
    
    auc_roc = roc_auc_score(y_true, scores)
    auc_pr = average_precision_score(y_true, scores)
        
    roc_all.append(auc_roc)
    pr_all.append(auc_pr)
    
    data = {'ROC': (roc_all),'PR': (pr_all)}
        
    df2 = pd.DataFrame(data)
    
        # #export to csv
    df2.to_csv('./{}/summary_results_{}.csv'.format(dir_, mode), index=False, encoding='utf-8-sig')
    