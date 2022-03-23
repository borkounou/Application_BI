
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import confusion_matrix, classification_report

from data_preprocess import data, main_data_splitter
x_train, x_valid,x_test, y_train, y_valid,y_test= main_data_splitter(data,numerical=True, ignored_pledged=True)

EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 0.0007
NUM_FEATURES = len(x_train.columns)
NUM_CLASSES = 5

class ClassifierDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


x_train = np.array(x_train)
x_valid = np.array(x_valid)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_valid = np.array(y_valid)
y_test = np.array(y_test)
train_dataset = ClassifierDataset(torch.from_numpy(x_train).float(),torch.from_numpy(y_train).long())

test_dataset = ClassifierDataset(torch.from_numpy(x_test).float(),torch.from_numpy(y_test).long())
val_dataset = ClassifierDataset(torch.from_numpy(x_valid).float(),torch.from_numpy(y_valid).long())





train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
val_loader = DataLoader(dataset=val_dataset, batch_size=1)
test_loader = DataLoader(dataset=test_dataset, batch_size=1)


class NeuralClassifier(nn.Module):
    def __init__(self, num_feature, num_class):
        super(NeuralClassifier, self).__init__()

        self.layer_1 = nn.Linear(num_feature, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)

    
    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_out(x)
        
        return x



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



model = NeuralClassifier(num_feature = NUM_FEATURES, num_class=NUM_CLASSES)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
print(model)


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)
    
    return acc


accuracy_stats = {
    'train': [],
    "val": []
}

loss_stats = {
    'train': [],
    "val": []
}

def train_model():
    print("Begin training.")
    for e in range(1, EPOCHS+1):
        
        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()
            y_train_pred = model(X_train_batch)
            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = multi_acc(y_train_pred, y_train_batch)
            
            train_loss.backward()
            optimizer.step()      
            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()
            
            # Validation
            with torch.no_grad():
                val_epoch_loss = 0
                val_epoch_acc = 0
                model.eval()
                for X_val_batch, y_val_batch in val_loader:
                    X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                    y_val_pred = model(X_val_batch)       
                    val_loss = criterion(y_val_pred, y_val_batch)
                    val_acc = multi_acc(y_val_pred, y_val_batch)
                    val_epoch_loss += val_loss.item()
                    val_epoch_acc += val_acc.item()

            loss_stats['train'].append(train_epoch_loss/len(train_loader))
            loss_stats['val'].append(val_epoch_loss/len(val_loader))
            accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
            accuracy_stats['val'].append(val_epoch_acc/len(val_loader))

            print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')


    # Create dataframes
    train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})# Plot the dataframes
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,7))
    sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
    sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')


    y_pred_list = []
    with torch.no_grad():
        model.eval()
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            _, y_pred_tags = torch.max(y_test_pred, dim = 1)
            y_pred_list.append(y_pred_tags.cpu().numpy())
            y_pred_list = [a for a in y_pred_list]


    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred_list))
    sns.heatmap(confusion_matrix_df, annot=True)

    print(classification_report(y_test, y_pred_list))

#https://towardsdatascience.com/pytorch-tabular-multiclass-classification-9f8211a123ab
#https://medium.com/analytics-vidhya/basic-of-correlations-and-using-pandas-and-scipy-for-calculating-correlations-2d16c2bd6af0
# https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/
#https://scikit-learn.org/stable/modules/svm.html
# https://www.journaldunet.fr/web-tech/guide-de-l-intelligence-artificielle/1501879-machine-a-vecteurs-de-support-svm-definition-et-cas-d-usage/
# https://medium.com/analytics-vidhya/what-is-the-use-of-data-standardization-and-where-do-we-use-it-in-machine-learning-97b71a294e24
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html
# https://blog.ineuron.ai/Categorical-Naive-Bayes-Classifier-implementation-in-Python-dAVqLWkf7E
