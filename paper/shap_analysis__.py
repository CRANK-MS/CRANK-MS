import pandas as pd
import joblib
import torch
import os
import shap


torch.set_grad_enabled(False)

# select model
model_name = 'svm'
assert model_name in ['mlp', 'xgb', 'rf', 'lr', 'svm','lda']

# data source csv
data_csv = './data/test crank-ms.csv'
dset_dir = os.path.split(data_csv)[-1][:-4]
root = 'test crank-ms'

#create data and values folder
save_dir = './{}/{}/{}'.format(root, dset_dir, model_name)
os.makedirs(save_dir, exist_ok=True)
os.makedirs(save_dir + '/data', exist_ok=True)
os.makedirs(save_dir + '/values', exist_ok=True)

# read data
df = pd.read_csv(data_csv)
x = df.iloc[:,1:].to_numpy()
y = df.iloc[:,0].to_numpy()


# load all model checkpoints one by one and apply SHAP analysis
ckpt_filepath = './{}/checkpoints'.format(save_dir)
for ckpt_filename in os.listdir(ckpt_filepath):
    # load model
    model = joblib.load('{}/{}'.format(ckpt_filepath, ckpt_filename))

    # normalize x
    x_ = model.norm_obj_.transform(x)
    
    # run shap
    explainer = shap.PermutationExplainer(model.predict, x_, max_evals=2**12)
    shap_values = explainer(x_)

    # plot shap figure
    # shap.plots.bar(shap_values, max_display = 20)
    # shap.summary_plot(shap_values)
    
    feature_values = pd.DataFrame(list(shap_values.values))
    feature_data = pd.DataFrame(list(shap_values.data))

    #save to csv
    feature_values.to_csv('./{}/{}/{}/values/{}_values.csv'.format(root, dset_dir, model_name, ckpt_filename), index=False)
    feature_data.to_csv('./{}/{}/{}/data/{}_data.csv'.format(root, dset_dir, model_name, ckpt_filename), index=False)

