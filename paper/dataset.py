import torch
import csv
import numpy as np

class Data(torch.utils.data.Dataset):

    def __init__(self, label, features, csv_dir):

        self.features = features
        self.label = label
        content = self.read_csv(csv_dir)
        self.content = self.filter_incomplete_cases(content)

        self.x = [[row[k] for k in self.features] for row in self.content]
        self.y = [row[self.label] for row in self.content]
        self.x = np.array(self.x, dtype = np.float32)
        self.y = np.array(self.y, dtype = np.float32)

    def read_csv(self, csv_file):

        content = []
        with open(csv_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                content.append(row)
        return content
    
    def filter_incomplete_cases(self, content):
        
        filtered_content = []
        for row in content:
            complete = True
            for key in self.features:
                if row[key] == '':
                    complete = False
            if complete and row[self.label] != '':
                filtered_content.append(row)
        return filtered_content

    def __len__(self):

        return len(self.content)

    def __getitem__(self, idx):

        return self.x[idx], self.y[idx]

    def input_length(self):

        return len(self.__getitem__(0)[0])
    
    @property
    def all(self):
        
        return self.x, self.y


if __name__ == "__main__":
    
    import pandas as pd

    df = pd.read_csv('./data/PD LCMS pos.csv', encoding='latin1')
    data = Data(
        label = 'PD',
        features = df.iloc[:, 1:-1],
        csv_dir = './data/PD LCMS pos.csv',
    )
    
    # dataloader = DataLoader(data,
    #                         batch_size=4,
    #                         shuffle=True)
    
    
    # for batch_input, batch_label in dataloader:
    #     # print(batch_input.shape, batch_label.shape)
    #     print(batch_input)
    #     print(batch_label)
