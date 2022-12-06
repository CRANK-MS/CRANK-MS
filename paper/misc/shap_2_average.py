import pandas as pd

root = 'test crank-ms'
data = 'test crank-ms'
mode = 'svm'

dir_ = './{}/{}/{}'.format(root, data, mode)

df = pd.read_csv('{}/values/combined_values.csv'.format(dir_))
df = df.abs()

average = df.mean(axis=0)
# average_sort = average.sort_values(ascending=False)
average.to_csv('./{}/shap absolute average.csv'.format(dir_), header = ['shap value'])