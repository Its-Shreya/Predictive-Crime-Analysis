import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear
from data_load import *
import argparse
import json
import global_var
import time
from sklearn.metrics import r2_score
from utils import *
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import global_var
import pandas as pd
import seaborn as sns
%matplotlib inline

# Load dataset
crime_data = pd.read_csv("crime_dataset.csv")  # Replace "crime_dataset.csv" with your dataset path
data=pd.read_csv('data.csv')
# Data Cleaning (if necessary)
# Check for missing values and handle them
crime_data.dropna(inplace=True)

# Feature Encoding (if necessary)
# Convert categorical variables into numerical representations using one-hot encoding or label encoding
crime_data = pd.get_dummies(crime_data, columns=['crime_location'])

# Separate features (X) and target variable (y)
X = crime_data.drop(columns=['crime_type'])
y = crime_data['crime_type']
# Feature Selection (if necessary)
# Use correlation analysis or feature importance from tree-based models to select relevant features
# using correlation analysis
correlation_matrix = crime_data.corr()
relevant_features = correlation_matrix[correlation_matrix['crime_type'] > 0.1].index.tolist()
X = crime_data[relevant_features]
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, classification_report

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
# Prediction
predicted_hotspots = model.predict(new_data)

# Display predicted hotspots
print("Predicted Crime Hotspots:", predicted_hotspots)

for col in data:
    print (type(data[col][1]))
data['timestamp'] = pd.to_datetime(data['timestamp'], coerce=True)
data['timestamp'] = pd.to_datetime(data['timestamp'], format = '%d/%m/%Y %H:%M:%S')
data['timestamp']
# DATE TIME STAMP FUNCTION
column_1 = data.ix[:,0]

db=pd.DataFrame({"year": column_1.dt.year,
              "month": column_1.dt.month,
              "day": column_1.dt.day,
              "hour": column_1.dt.hour,
              "dayofyear": column_1.dt.dayofyear,
              "week": column_1.dt.week,
              "weekofyear": column_1.dt.weekofyear,
              "dayofweek": column_1.dt.dayofweek,
              "weekday": column_1.dt.weekday,
              "quarter": column_1.dt.quarter,
             })
dataset1=dataset.drop('timestamp',axis=1)
data1=pd.concat([db,dataset1],axis=1)
data1.info()
sns.pairplot(data1,hue='act363')
sns.boxplot(x='act379' ,y='hour' ,data=data1, palette='winter_r')
sns.boxplot(x='act13' ,y='hour' ,data=data1 , palette='winter_r')
sns.boxplot(x='act323' ,y='hour' ,data=data1, palette='winter_r')
sns.boxplot(x='act363' ,y='hour' ,data=data1, palette='winter_r')
df = pd.DataFrame(data=data1, columns=['act13', 'hour', 'day'])
df.plot.hexbin(x='act13',y='hour',gridsize=25)
df.plot(legend=False)
df1 = pd.DataFrame(data=data1, columns=['act13', 'act323', 'act379'])
df1.plot.kde()
X=data1.iloc[:,[1,2,3,4,6,16,17]].values

y=data1.iloc[:,[10,11,12,13,14,15]].values
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=50)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train)
knn.score(X_test,y_test)
knn.score(X_train,y_train)
error_rate = []
for i in range(1,140):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))
plt.plot(range(1,140),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=5)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(max_depth=500, random_state=300)
dtree.fit(X_train,y_train)
y_pred=dtree.predict(X_test)
dtree.score(X_test,y_test)
dtree.score(X_train,y_train)
treefeatures=dtree.feature_importances_
indices = np.argsort(treefeatures)

features = data1.iloc[:,[1,2,3,4,6,16,17]]
plt.figure(1)
plt.title('Feature Importances')
plt.barh(range(len(indices)), treefeatures[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')

feature_names=[ 'dayofweek', 'dayofyear', 'hour', 'month', 'week','latitude', 'longitude']
from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydot 
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
dot_data = StringIO()  
export_graphviz(dtree, out_file=dot_data,feature_names=feature_names,filled=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png())  
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
y_pred=rfc.predict(X_test)
rfc.score(X_test,y_test)
rfc.score(X_train,y_train)
imp=rfc.feature_importances_
indices = np.argsort(imp)
features = data1.columns
plt.figure(1)
plt.title('Feature Importances')
plt.barh(range(len(indices)), om[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')

config = global_var.get_value('config')


class BasicBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialization=True):
        super(BasicBlock, self).__init__()
        self.initialization = initialization
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False, dilation=config['DILATION'])

        if initialization:
            self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.conv.weight)

    def forward(self, x):
        return self.conv(x)

class GatingBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialization=True, activation=None):
        super(GatingBlock, self).__init__()

        self.activation = activation

        self.block1 = BasicBlock(in_channels, out_channels, kernel_size, stride, padding, initialization)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.block2 = BasicBlock(in_channels, out_channels, kernel_size, stride, padding, initialization)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out1 = self.block1(x)
        out1 = torch.sigmoid(self.bn1(out1))
        out2 = self.block2(x)
        out2 = self.bn2(out2)
        if self.activation != None:
            out2 = self.activation(out2)
        return out1 * out2


class TimeSequenceModel(torch.nn.Module):
    def __init__(self, activation=torch.tanh):
        super().__init__()
        self.gcnns = torch.nn.Sequential(GatingBlock(1, 1, 3,  activation=activation),
                                        GatingBlock(1, 1, 3, activation=activation))
        self.fc = nn.Linear(config['TIME_SERIES_LENGTH']-2*2,1)

        # test new model
        # self.fcnew1 = nn.Linear(config['TIME_SERIES_LENGTH'],1)
        # self.fcnew2 = nn.Linear(config['TIME_SERIES_LENGTH'], 1)
        # self.fcnew3 = nn.Linear(config['TIME_SERIES_LENGTH'], 1)
        # end test

    # input shape: (batch_size=area, channels=1, TIME_SERIES_LENGTH)
    # output shape: (area, 1, 1)
    def forward(self, x):
        out = self.gcnns(x)
        out = self.fc(out)
        return out
        # out = self.fcnew1(x) + self.fcnew1(x) + self.fcnew3(x)
        return out


class LinearAggregation(torch.nn.Module):
    def __init__(self, activation=torch.tanh):
        super().__init__()
        self.fc1 = nn.Linear(config['NUM_CLASS_CRIME']+config['NUM_CLASS_311']+config['NUM_CLASS_POI'],config['DIM_NODE_FEATURE'])
        # self.fc1 = nn.Linear(config['NUM_CLASS_CRIME'] , config['DIM_NODE_FEATURE'])
        # self.fc2 = nn.Linear(config['DIM_NODE_FEATURE'], config['DIM_NODE_FEATURE'])
    # input shape: (NUM_DAYS, NUM_AREAS, NUM_CLASS_CRIME+NUM_CLASS_311+NUM_CLASS_POI)
    # output shape: (NUM_DAYS, NUM_AREAS, DIM_NODE_FEATURE)
    def forward(self, x):
        # x = x[:,:,0:config['NUM_CLASS_CRIME']]
        out = self.fc1(x)
        # out = self.fc2(out)
        return out


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('area', 'bike', 'area'): GATConv(-1, hidden_channels, heads=config['NUM_HEADS'], concat=False, dropout=0.5),
                ('area', 'taxi', 'area'): GATConv(-1, hidden_channels, heads=config['NUM_HEADS'], concat=False, dropout=0.5),
                ('area', 'geo', 'area'): GATConv(-1, hidden_channels, heads=config['NUM_HEADS'], concat=False, dropout=0.5),
                ('area', 'simi', 'area'): GATConv(-1, hidden_channels, heads=config['NUM_HEADS'], concat=False, dropout=0.5)
            }, aggr='mean')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    # x_dict shape: (263,1)
    # edge_index_dict shape: (263,263)
    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return self.lin(x_dict['area'])


class STHGNN(torch.nn.Module):
    def __init__(self, time_series_length, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.lin_agg = LinearAggregation()
        self.het_gnns = torch.nn.ModuleList()
        for _ in range(time_series_length):
            het_gnn = HeteroGNN(hidden_channels, out_channels, num_layers)
            self.het_gnns.append(het_gnn)
        self.t_model = TimeSequenceModel()
        self.lin1 = Linear(263, 100)
        self.lin2 = Linear(100, 263)
        # self.tpdata = torch.rand((7,263,1))

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='configuration file')
parser.add_argument("--output", type=str, help='output path')
args = parser.parse_args()

config_filename = args.config
output_path = args.output


with open(config_filename, 'r') as f:
    config = json.loads(f.read())

seed = config['seed']
setup_seed(seed)

global_var._init()
global_var.set_value('config',config)

from model import STHGNN
from model_NO_N import STHGNN_N
from model_NO_S import STHGNN_S
from model_NO_T import STHGNN_T
from model_NO_M import STHGNN_noM
from model_NO_F import STHGNN_noF
from model_NO_EMF import STHGNN_c_g


from data_load import *


def train(model, train_data_loader, optimizer):
    model.train()
    # loss_fn = nn.MSELoss()
    loss_fn = nn.HuberLoss(reduction='mean', delta=config['DELTA'])
    st = time.time()
    output_list = []
    label_list = []
    for i, input_data in enumerate(train_data_loader):
        optimizer.zero_grad()
        input_data = input_data.to(device)

        label = input_data.y
        out = model(input_data)

        loss = loss_fn(out, label)

        pred = np.floor(out.cpu().reshape(263).detach().numpy())
        y_true = label.cpu().reshape(263).detach().numpy()
        output_list.append(pred)
        label_list.append(y_true)

        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()
        if i%50 == 0:
            print(f'{time.time()-st:.2f}: loss: {loss}.  {i}/{len(train_data_loader)}')

    output = np.array(output_list)
    label = np.array(label_list)

    train_mse_loss = mse_np(output, label)
    train_rmse_loss = rmse_np(output, label)
    train_mae_loss = mae_np(output, label)
    train_mape_loss = mape_np(output, label)
    train_r2_loss = r2_score(output.reshape((-1, 1)), label.reshape((-1, 1)))

    return train_mse_loss, train_rmse_loss, train_mae_loss, train_mape_loss, train_r2_loss


def test(model, test_data_loader):
    model.eval()
    output_list = []
    label_list = []
    with torch.no_grad():
        for i, input_data in enumerate(test_data_loader):
            input_data = input_data.to(device)
            label = input_data.y
            out = model(input_data)

            pred = np.floor(out.cpu().reshape(263).numpy())
            y_true = label.cpu().reshape(263).numpy()

            output_list.append(pred)
            label_list.append(y_true)

            torch.cuda.empty_cache()
            if i % 50 == 0:
                print(f'testing...  {i}/{len(test_data_loader)}')

        output = np.array(output_list)
        label = np.array(label_list)

        test_mse_loss = mse_np(output,label)
        test_rmse_loss = rmse_np(output,label)
        test_mae_loss = mae_np(output,label)
        test_mape_loss = mape_np(output,label)
        test_r2_loss = r2_score(np.array(output_list).reshape((-1,1)), np.array(label_list).reshape((-1,1)))

    return test_mse_loss, test_rmse_loss, test_mae_loss, test_mape_loss, test_r2_loss


best_rmse_list = []
best_mae_list = []
best_mape_list = []
for r in range(config['REPEAT_TIMES']):
    plt.cla()
    path = output_path + 'times_{}'.format(r)
    if not os.path.exists(path):
        os.mkdir(path)

    # setup_seed(20+r)
    train_data_loader, test_data_loader = load_data(config['CRIME_LABEL_DATA_PATH'], config['CRIME_DATA_PATH'],
                                                    config['A311_DATA_PATH'],config['POI_DATA_PATH'],
                                                    config['TAXI_DATA_PATH'], config['BIKE_DATA_PATH'],
                                                    config['GEO_DATA_PATH'],config['SIMI_DATA_PATH'], config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")

    if 'WITHOUT_N' in config and config['WITHOUT_N'] == 1:
        model = STHGNN_N(time_series_length=config['TIME_SERIES_LENGTH'], hidden_channels=config['HIDDEN_CHANNELS'],
                       out_channels=1, num_layers=config['NUM_LAYERS'])
    # elif 'WITHOUT_E' in config and config['WITHOUT_E'] == 1:
    #     model = STHGNN_E(time_series_length=config['TIME_SERIES_LENGTH'], hidden_channels=config['HIDDEN_CHANNELS'],
    #                      out_channels=1, num_layers=config['NUM_LAYERS'])
    elif 'WITHOUT_S' in config and config['WITHOUT_S'] == 1:
        model = STHGNN_S(time_series_length=config['TIME_SERIES_LENGTH'], hidden_channels=config['HIDDEN_CHANNELS'],
                         out_channels=1, num_layers=config['NUM_LAYERS'])
    elif 'WITHOUT_T' in config and config['WITHOUT_T'] == 1:
        model = STHGNN_T(time_series_length=config['TIME_SERIES_LENGTH'], hidden_channels=config['HIDDEN_CHANNELS'],
                         out_channels=1, num_layers=config['NUM_LAYERS'])
    elif 'WITHOUT_M' in config and config['WITHOUT_M'] == 1:
        model = STHGNN_noM(time_series_length=config['TIME_SERIES_LENGTH'], hidden_channels=config['HIDDEN_CHANNELS'],
                         out_channels=1, num_layers=config['NUM_LAYERS'])
    elif 'WITHOUT_F' in config and config['WITHOUT_F'] == 1:
        model = STHGNN_noF(time_series_length=config['TIME_SERIES_LENGTH'], hidden_channels=config['HIDDEN_CHANNELS'],
                         out_channels=1, num_layers=config['NUM_LAYERS'])
    elif 'CRIME_DATA' in config:
        if config['CRIME_DATA']==1 and config['POI_DATA']==0 and config['311_DATA']==0 and config['GEO_DATA']==1 \
                and config['TAXI_DATA']==0 and config['BIKE_DATA']==0 and config['SIMI_DATA']==0:
            model = STHGNN_c_g(time_series_length=config['TIME_SERIES_LENGTH'],
                               hidden_channels=config['HIDDEN_CHANNELS'],
                               out_channels=1, num_layers=config['NUM_LAYERS'])
        elif config['CRIME_DATA']==1 and config['POI_DATA']==1 and config['311_DATA']==1 and config['GEO_DATA']==1 \
                and config['TAXI_DATA']==1 and config['BIKE_DATA']==1 and config['SIMI_DATA']==1:
            model = STHGNN(time_series_length=config['TIME_SERIES_LENGTH'],
                               hidden_channels=config['HIDDEN_CHANNELS'],
                               out_channels=1, num_layers=config['NUM_LAYERS'])
    else:
        print('invalid config!')
        break
        # model = STHGNN(time_series_length=config['TIME_SERIES_LENGTH'], hidden_channels=config['HIDDEN_CHANNELS'],
        #                out_channels=1, num_layers=config['NUM_LAYERS'])
    model = model.to(device)
    # model = torch.load('/home/zh/temp/new_20nodefeatures_threshold5/model_10_epoch.pth')

    optimizer = torch.optim.Adam(model.parameters(), lr=config['LEARNING_RATE'], weight_decay=config['WEIGHT_DECAY'])

    train_mse_loss_list = []
    train_rmse_loss_list = []
    train_mae_loss_list = []
    train_mape_loss_list = []
    train_r2_loss_list = []

    test_mse_loss_list = []
    test_rmse_loss_list = []
    test_mae_loss_list = []
    test_mape_loss_list = []
    test_r2_loss_list = []

    best = []
    min_rmse = 1000000000.0
    for epoch in range(0, config['EPOCH']):
        start_time = time.time()
        train_mse_loss, train_rmse_loss, train_mae_loss, train_mape_loss, train_r2_loss = train(model, train_data_loader, optimizer)
        test_mse_loss, test_rmse_loss, test_mae_loss, test_mape_loss, test_r2_loss = test(model, test_data_loader)

        if test_rmse_loss < min_rmse:
            min_rmse = test_rmse_loss
            best = [test_mse_loss, test_rmse_loss, test_mae_loss, test_mape_loss, test_r2_loss, epoch]

        train_mse_loss_list.append(train_mse_loss)
        train_rmse_loss_list.append(train_rmse_loss)
        train_mae_loss_list.append(train_mae_loss)
        train_mape_loss_list.append(train_mape_loss)
        train_r2_loss_list.append(train_r2_loss)

        test_mse_loss_list.append(test_mse_loss)
        test_rmse_loss_list.append(test_rmse_loss)
        test_mae_loss_list.append(test_mae_loss)
        test_mape_loss_list.append(test_mape_loss)
        test_r2_loss_list.append(test_r2_loss)

        print(f'{time.time()-start_time:.2f}: Epoch: {epoch:03d}, Train loss: '
              f' MAE {train_mae_loss:.4f};  MAPE {train_mape_loss:.4f}; R2 {train_r2_loss:.4f}')
        print(f'{time.time()-start_time:.2f}: Epoch: {epoch:03d}, Test loss: '
              f' MAE {test_mae_loss:.4f};  MAPE {test_mape_loss:.4f}; R2 {test_r2_loss:.4f}')
        print(f'{time.time() - start_time:.2f}:Best Epoch: {best[5]:03d}, '
              f' MAE {best[2]:.4f};  MAPE {best[3]:.4f}; R2 {best[4]:.4f}')
        if epoch%10==0:
            torch.save(model, path + '/' +'model_{}_epoch.pth'.format(epoch))

    plt.plot(train_rmse_loss_list, label="train rmse")
    plt.plot(test_rmse_loss_list, label="test rmse")
    plt.legend()
    plt.savefig(path + '/' + 'figure.jpg')
    plt.show()

    best_mae_list.append(best[2])
    best_mape_list.append(best[3])
    print('best tesing results: MAE: {:.2f}\ntesting: RMSE: {:.2f}\ntesting: MAPE: {:.2f}\n'.format(best[2], best[1], best[3]))


save_variable(best_mae_list,output_path + 'best_mae_list')
save_variable(best_mape_list,output_path + 'best_mape_list')
print('best MAE list:')
print(best_mae_list)
print('best MAPE list:')
print(best_mape_list)
print('final results mean: MAE: {:.2f}\ntesting: MAPE: {:.2f}\n'.format(np.mean(best_mae_list),np.mean(best_mape_list)))
print('final results range: MAE: {:.2f}\ntesting: MAPE: {:.2f}\n'
      .format((np.max(best_mae_list)-np.min(best_mae_list)),
              (np.max(best_mape_list)-np.min(best_mape_list))))


    # hst_data is a HSTData object
    def forward(self, hst_data):
        the_data = hst_data.hetero_data_list[0]

        #   node feature aggregation
        x_temp = torch.zeros((config['TIME_SERIES_LENGTH'], config['NUM_AREAS'],
                              config['NUM_CLASS_CRIME']+config['NUM_CLASS_311']+config['NUM_CLASS_POI'])).to('cuda')
        for i in range(config['TIME_SERIES_LENGTH']):
            x_temp[i,:,:] = torch.cat((the_data[i]['crime'].x, the_data[i]['a311'].x, the_data[i]['poi'].x), 1)

        agg_results = self.lin_agg(x_temp)
        for i in range(config['TIME_SERIES_LENGTH']):
            the_data[i]['area'].x = agg_results[i,:,:]

        #   GNN模型
        X = torch.zeros((config['TIME_SERIES_LENGTH'], config['NUM_AREAS'], 1)).to('cuda')
        for i in range(len(self.het_gnns)):
            het_gnn = self.het_gnns[i]
            gpu_data = the_data[i]
            x_t = het_gnn(gpu_data.x_dict, gpu_data.edge_index_dict)
            X[i,:,:] = x_t

        #   时序模型
        a = X.permute(1,2,0)
        b = self.t_model(a)
        c = b.reshape(1,1,263)
        d = self.lin1(c)
        e = self.lin2(d)
        f = e.reshape(263,1,1)
        return f

