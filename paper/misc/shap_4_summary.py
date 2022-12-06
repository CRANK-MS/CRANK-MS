import pandas as pd

root = 'test crank-ms'
data = 'test crank-ms'
mode = 'svm'

dir_ = './{}/{}/{}'.format(root, data, mode)

df_values = pd.read_csv('{}/shap absolute average.csv'.format(dir_))
df_corr = pd.read_csv('./{}/shap correlation.csv'.format(dir_))

df_merge = pd.merge(df_values, df_corr)
average_sort = df_merge.sort_values(by='shap value', ascending=False)
average_sort.to_csv('./{}/shap summary.csv'.format(dir_))