import pandas as pd
import glob
import os

root = 'test crank-ms'
data = 'test crank-ms'
mode = 'svm'
folder = 'data' #values or data

dir_ = './{}/{}/{}/{}'.format(root, data, mode, folder)

os.chdir(dir_)

extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])

#export to csv
combined_csv.to_csv( "combined_{}.csv".format(folder), index=False, encoding='utf-8-sig')


