import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as sf
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import DoubleType, ArrayType

from sim4rec.recommenders.ucb import UCB

import sklearn 
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sim4rec.utils import pandas_to_spark

class DecisionTreeRecommender():
    def __init__(self, seed=None):
        self.seed = seed
        np.random.seed(seed)
        self.model = DecisionTreeRegressor(random_state=self.seed)
        self.scalar = StandardScaler()
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

            pd_log = pd.get_dummies(pd_log)
            pd_log['price'] = self.scalar.fit_transform(pd_log[['price']])

            y = pd_log['relevance']
            x = pd_log.drop(['relevance'], axis=1)

            self.model.fit(x,y)
            self.trained = True

    def predict(self, log, k, users:DataFrame, items:DataFrame, user_features=None, item_features=None, filter_seen_items=True):

        cross = users.join(items).drop('__iter').toPandas().copy()
        
        cross = pd.get_dummies(cross)
        cross['orig_price'] = cross['price']
        cross['price'] = self.scalar.transform(cross[['price']])        
        

        cross['prob'] = self.model.predict(cross.drop(['user_idx', 'item_idx', 'orig_price'], axis=1))

        cross['relevance'] = cross['prob'] + 0.5 * cross['price']

        cross = cross.sort_values(by=['user_idx', 'relevance'], ascending=[True, False])
        cross = cross.groupby('user_idx').head(k)

        cross['price'] = cross['orig_price']
       
        return pandas_to_spark(cross)
        