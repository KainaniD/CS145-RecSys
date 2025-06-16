import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sim4rec.utils import pandas_to_spark

# Suppress fragmentation warnings
import warnings
warnings.filterwarnings("ignore")

class ARRecommender():
    def __init__(self, max_lag=1):
        self.max_lag = max_lag
        
        self.model = LogisticRegression()
        self.scaler = StandardScaler()
        self.trained = False
        
    def fit(self, log, user_features=None, item_features=None):
        # Merge transaction log, user features, and item features
        pd_log = log.join(
            user_features,
            on='user_idx'
        ).join(
            item_features,
            on='item_idx'
        ).toPandas()
        
        # Sort items by user index, then iteration
        if self.trained:
            pd_log = pd_log.sort_values(by=['user_idx','__iter'], ignore_index=True)
            pd_log = pd_log.drop(columns=['user_idx','item_idx','__iter'])
        else:
            pd_log = pd_log.sort_values(by='user_idx', ignore_index=True)
            pd_log = pd_log.drop(columns=['user_idx','item_idx'])
        
        # Scale price
        pd_log['scaled_price'] = self.scaler.fit_transform(pd_log[['price']])
        
        # Create lagged values of features
        for lag in range(1,self.max_lag+1):
            for i in range(20):
                pd_log[f'user_attr_{i}_{lag}'] = pd_log[f'user_attr_{i}'].shift(lag, fill_value=0)
            for i in range(20):
                pd_log[f'item_attr_{i}_{lag}'] = pd_log[f'item_attr_{i}'].shift(lag, fill_value=0)
            pd_log[f'scaled_price_{lag}'] = pd_log['scaled_price'].shift(lag, fill_value=0)
            pd_log[f'category_{lag}'] = pd_log['category'].shift(lag)
            pd_log[f'segment_{lag}'] = pd_log['segment'].shift(lag)
        # Get one-hot encodings for categorical features (automatically zero for NaN)
        pd_log = pd.get_dummies(pd_log)
        
        # Create features and label for the model
        X = pd_log.drop(columns=['relevance','price'])
        y = pd_log['relevance']
        
        self.model.fit(X,y)
        self.trained = True
    
    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        # Cross-join users and items
        users_pd = users.toPandas()
        items_pd = items.toPandas()
        cross = users_pd.join(items_pd, how='cross')
        
        # Scale price
        cross['scaled_price'] = self.scaler.transform(cross[['price']])
        
        # Create lagged values of features
        for lag in range(1,self.max_lag+1):
            for i in range(20):
                cross[f'user_attr_{i}_{lag}'] = cross[f'user_attr_{i}'].shift(lag, fill_value=0)
            for i in range(20):
                cross[f'item_attr_{i}_{lag}'] = cross[f'item_attr_{i}'].shift(lag, fill_value=0)
            cross[f'scaled_price_{lag}'] = cross['scaled_price'].shift(lag, fill_value=0)
            cross[f'category_{lag}'] = cross['category'].shift(lag)
            cross[f'segment_{lag}'] = cross['segment'].shift(lag)   
        # Get one-hot encodings for categorical features (automatically zero for NaN)
        cross = pd.get_dummies(cross)
        
        # Get predictions from logistic regression model
        X = cross.drop(columns=['user_idx','item_idx','price'])
        cross['relevance'] = self.model.predict_proba(X)[:,1]
        
        # Calculate recommendation score as relevance * price
        cross['score'] = cross['relevance'] * cross['price']
        
        # Group recommendations by user and keep only the best k
        cross = cross.sort_values(by=['user_idx', 'score'], ascending=[True, False])
        cross = cross.groupby('user_idx').head(k)
        
        # Return results as a Spark DataFrame
        return pandas_to_spark(cross[['user_idx', 'item_idx', 'relevance']])