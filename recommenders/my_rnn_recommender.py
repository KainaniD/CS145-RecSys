import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from pyspark.sql import DataFrame
from sim4rec.utils import pandas_to_spark

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=0.3 if layer_dim > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h0=None):
        if h0 is None:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)

        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out, hn

class RNNRecommender():
    def __init__(self, seed=None, epoch_num=20, hidden_dim=32, layer_dim=2, lr=0.01):
        self.seed = seed
        np.random.seed(seed)

        self.model = RNNModel(input_dim=1, hidden_dim=hidden_dim, layer_dim=layer_dim, output_dim=1)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.scalar = StandardScaler()
        self.epoch_num = epoch_num
        self.trained = False

    def fit(self, log: DataFrame, user_features=None, item_features=None):
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

            pd_log = pd.get_dummies(pd_log)
            pd_log['price'] = self.scalar.fit_transform(pd_log[['price']])

            y = pd_log['relevance']
            x = pd_log.drop(['relevance'], axis=1)
            x = x.astype(np.float32)
            y = y.astype(np.float32)
            x = torch.from_numpy(x.values).unsqueeze(2)
            y = torch.from_numpy(y.values).unsqueeze(1)

            h0 = None
            for epoch in range(self.epoch_num):
                self.model.train()
                self.optimizer.zero_grad()
                outputs, h0 = self.model(x, h0)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                h0 = h0.detach()

            self.trained = True

    def predict(self, log, k, users: DataFrame, items: DataFrame, user_features=None, item_features=None, filter_seen_items=True):
        cross = users.join(items).drop('__iter').toPandas().copy()
        cross = pd.get_dummies(cross)
        cross['orig_price'] = cross['price']
        cross['price'] = self.scalar.transform(cross[['price']])

        cross32 = cross.astype(np.float32)
        cross_tensor = torch.from_numpy(cross32.values).unsqueeze(2)

        with torch.no_grad():
            prob, _ = self.model(cross_tensor)

        cross['prob'] = torch.sigmoid(prob).squeeze().numpy()
        cross['relevance'] = cross['prob'] * cross['price']

        cross = cross.sort_values(by=['user_idx', 'relevance'], ascending=[True, False])
        cross = cross.groupby('user_idx').head(k)
        cross['price'] = cross['orig_price']
        cross = cross.astype({col: 'int32' for col in cross.select_dtypes(include=['uint8']).columns})

        return pandas_to_spark(cross)
