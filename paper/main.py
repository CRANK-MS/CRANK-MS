import os, shutil
import matplotlib.pyplot as plt
import pandas as pd
from dataset import Data
from misc import get_metrics
from train import train
from bootstrap import bootstrap
from plot import plot_roc, plot_pr


# select model
model_name = 'svm'
assert model_name in ['mlp', 'xgb', 'rf', 'lr', 'svm','lda']

# data source csv
data_csv = './data/test crank-ms.csv'

# directory to save results
'''
combined_outputs/
    PD LCMS pos/            (dataset specific)
        mlp/                (model specific)
            checkpoints/    (to save trained models)
                test_0.*
                test_1.*
                ...
            results/        (to save labels, scores, predictions etc.)
                test_0.csv
                test_1.csv
                ...
        ...
    ...
'''
comb_dir = './test crank-ms'
dset_dir = os.path.split(data_csv)[-1][:-4]
mode_dir = model_name
save_dir = '{}/{}/{}'.format(comb_dir, dset_dir, mode_dir)

# clean up the result folder
shutil.rmtree(save_dir, ignore_errors=True)
os.makedirs(save_dir, exist_ok=True)
os.makedirs(save_dir + '/checkpoints', exist_ok=True)
os.makedirs(save_dir + '/results', exist_ok=True)

# initialize dataset
df = pd.read_csv(data_csv, encoding = 'latin1')
whole_data = Data(
    label = 'PD',
    features = df.iloc[:,1:],
    csv_dir = data_csv,
)

# for random data split
train_size = int(len(whole_data) * 0.6)
valid_size = len(whole_data) - train_size

# run for multiple times to collect the statistics of the performance metrics
bootstrap(
    func = train,
    args = (model_name, whole_data, train_size, valid_size, save_dir),
    kwargs = {},
    num_runs = 100,
    num_jobs = 1,
)    

# calculate and show the mean & std of the metrics for all runs
list_csv = os.listdir(save_dir + '/results')
list_csv = ['{}/results/{}'.format(save_dir, fn) for fn in list_csv]

y_true_all, y_pred_all, scores_all = [], [], []
for fn in list_csv:
    df = pd.read_csv(fn)
    y_true_all.append(df['Y Label'].to_numpy())
    y_pred_all.append(df['Y Predicted'].to_numpy())
    scores_all.append(df['Predicted score'].to_numpy())

met_all = get_metrics(y_true_all, y_pred_all, scores_all)
for k, v in met_all.items():
    if k not in ['Confusion Matrix']:
        print('{}:\t{:.4f} \u00B1 {:.4f}'.format(k, v[0], v[1]).expandtabs(20))

# plot roc, pr curves
fig = plt.figure(figsize=(6, 6), dpi=100)
plot_roc(plt.gca(), y_true_all, scores_all)
fig.savefig('{}/ROC {}.png'.format(save_dir,model_name), dpi=300)

fig = plt.figure(figsize=(6, 6), dpi=100)
plot_pr(plt.gca(), y_true_all, scores_all)
fig.savefig('{}/PR {}.png'.format(save_dir, model_name), dpi=300) 
