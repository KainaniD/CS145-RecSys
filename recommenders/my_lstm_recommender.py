import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as sf
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import DoubleType, ArrayType

from sim4rec.recommenders.ucb import UCB

import torch
import torch.nn as nn
import torch.nn.functional as F

import sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sim4rec.utils import pandas_to_spark

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = F.sigmoid(self.fc(out[:, -1, :]))
        return out, hn, cn

class LSTMRecommender():
    def __init__(self, seed=None, epoch_num=10, hidden_dim=50, layer_dim=1, dropout=0.0):
        self.seed = seed
        np.random.seed(seed)

        self.model = LSTMModel(input_dim=1, hidden_dim=hidden_dim, layer_dim=layer_dim, output_dim=1, dropout=dropout)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.scalar = StandardScaler()
        self.epoch_num = epoch_num
        self.trained = False

    def fit(self, log:DataFrame, user_features=None, item_features=None):
        # log.show(5)
        # user_features.show(5)
        # item_features.show(5)

        if user_features and item_features:
            pd_log = log.join(
                user_features,
                on='user_idx'
            ).join(
                item_features,
                on='item_idx'
            ).drop(
                'user_idx', 'item_idx', '__iter'
            ).toPandas()

            #one-hot encode categorical features
            pd_log = pd.get_dummies(pd_log)
            pd_log['price'] = self.scalar.fit_transform(pd_log[['price']])

            #input and output variables
            y = pd_log['relevance']
            x = pd_log.drop(['relevance'], axis=1)
            x = x.astype(np.float32)
            y = y.astype(np.float32)
            x = torch.from_numpy(x.values).unsqueeze(2)
            y = torch.from_numpy(y.values).unsqueeze(1)

            #train model
            h0, c0 = None, None
            for epoch in range(self.epoch_num):
                self.model.train()
                self.optimizer.zero_grad()

                outputs, h0, c0 = self.model(x, h0, c0)

                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()

                h0 = h0.detach()
                c0 = c0.detach()

            self.trained = True



    def predict(self, log, k, users:DataFrame, items:DataFrame, user_features=None, item_features=None, filter_seen_items=True):

        cross = users.join(items).drop('__iter').toPandas().copy()
        #one hot encode categorical features
        cross = pd.get_dummies(cross)
        cross['orig_price'] = cross['price']
        cross['price'] = self.scalar.transform(cross[['price']])

        #making predictions using model

        cross32 = cross.astype(np.float32)
        cross_tensor = torch.from_numpy(cross32.values).unsqueeze(2)

        with torch.no_grad():
            prob, _, _ = self.model(cross_tensor)

        cross['prob'] = pd.DataFrame(prob.squeeze().numpy())


        #calculate relevance as prob * price
        cross['relevance'] = cross['prob'] * cross['price']
        
        #filter out seen items if required
        cross = cross.sort_values(by=['user_idx', 'relevance'], ascending=[True, False])
        cross = cross.groupby('user_idx').head(k)

        cross['price'] = cross['orig_price']

        cross = cross.astype({col: 'int32' for col in cross.select_dtypes(include=['uint8']).columns})
        return pandas_to_spark(cross)
