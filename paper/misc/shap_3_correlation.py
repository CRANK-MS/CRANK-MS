import pandas as pd
from scipy.stats import pearsonr
import csv
import numpy as np

root = 'test crank-ms'
data = 'test crank-ms'
mode = 'svm'

dir_ = './{}/{}/{}'.format(root, data, mode)

corr = []

for i in range(100):
    df1 = pd.read_csv('{}/values/test_{}.ckpt_values.csv'.format(dir_, i))
    df2 = pd.read_csv('{}/data/test_{}.ckpt_data.csv'.format(dir_, i))
    
    shap = df1.to_numpy()
    data = df2.to_numpy()
    
    # calculate Pearson's correlation
    corr.append(
        [pearsonr(shap[:,i], data[:,i])[0] for i in range(shap.shape[1])]
    )
    
corr = np.array(corr)
corr_mean = np.nanmean(corr, axis=0)
corr_mean= pd.DataFrame(corr_mean)
    
corr_mean.to_csv('./{}/shap correlation.csv'.format(dir_), header=['Corr'])
