import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LinearRegression

from sim4rec.utils import pandas_to_spark

class HybridAR():
    
    def __init__(self, max_lag=2, smoothing=1):
        self.max_lag = max_lag
        self.smoothing = smoothing
        self.scaler = StandardScaler()
        self.ordinal = OrdinalEncoder()
        
        # Matrix of category transition frequencies
        self.frequencies = np.zeros((4,4))
        
        # Column-stochastic Markov weight matrix
        self.markov = np.zeros((4,4))
        
        # Price autoregressor
        self.price_model = LinearRegression()
    
    def fit(self, log, user_features=None, item_features=None):
        
        # Construct merged transaction log with price and category
        items_pd = item_features.toPandas()[['item_idx','price','category']]
        log_pd = log.toPandas().merge(items_pd, on='item_idx')
        print(log_pd)
        
        # Apply ordinal encoding to category
        log_pd['category'] = self.ordinal.fit_transform(log_pd[['category']])
        log_pd['category'] = log_pd['category'].astype(int)
        
        # Scale price
        log_pd['scaled_price'] = self.scaler.fit_transform(log_pd[['price']])
        
        # Initialize the matrix of category transition frequencies to some nonzero smoothing factor
        # 1 will be added to each entry when the corresponding transition is observed
        self.frequencies = np.ones((4,4)) * self.smoothing
        
        # Array for holding price data to train the price autoregressor
        user_price_data = []
        
        # Iterate through each user in the transaction log
        unique_users = log_pd['user_idx'].unique()
        for user in unique_users:
            # Match the user in the transaction log and get the prices of all transactions
            user_log = log_pd[log_pd['user_idx']==user]
            all_user_prices = user_log['scaled_price'].to_numpy()
            
            # Truncate the price sequence at all possible points
            for i in range(len(all_user_prices)):
                trunc_user_prices = all_user_prices[:i+1]
                
                # Take the 'max_lag'+1 most recent transactions for the price autoregressor
                user_prices = trunc_user_prices[-(self.max_lag+1):]
                # Zero-pad this array if its length is less than 'max_lag'+1
                if len(user_prices) < self.max_lag+1:
                    user_prices = np.pad(
                        user_prices,
                        (self.max_lag+1-len(user_prices),0),
                        'constant',
                        constant_values=0
                    )
                user_price_data.append(user_prices)
            
            # Take the categories of all transactions for the Markov model
            user_categories = user_log['category'].to_numpy()
            # Add transitions to the category transition frequency matrix
            for i in range(1,len(user_categories)):
                self.frequencies[user_categories[i]][user_categories[i-1]] += 1
        
        # Convert user_price_data to numpy, and separate it into
        # features/lagged prices (X) and labels/current prices (y)
        user_price_data = np.array(user_price_data)
        user_price_X = user_price_data[:,:-1]
        user_price_y = user_price_data[:,-1]
        # print(user_price_X)
        # print(user_price_y)
        
        # Fit the autoregressor on this data
        self.price_model.fit(user_price_X, user_price_y)
        # print(self.price_model.coef_)
        # print(self.price_model.intercept_)
        
        # Generate the column-stochastic Markov weight matrix from the category transition frequency matrix
        self.markov = self.frequencies / self.frequencies.sum(axis=0)
        # print(self.markov)
    
    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        
        # Construct merged transaction log with price and category
        items_pd = item_features.toPandas()[['item_idx','price','category']]
        log_pd = log.toPandas().merge(items_pd, on='item_idx')
        
        # Apply ordinal encoding to category
        log_pd['category'] = self.ordinal.transform(log_pd[['category']])
        log_pd['category'] = log_pd['category'].astype(int)
        
        # Scale price
        log_pd['scaled_price'] = self.scaler.transform(log_pd[['price']])
        
        # Cross-join users and items; drop unused features
        users_pd = users.toPandas().drop(columns=['segment']+[f'user_attr_{i}' for i in range(20)])
        items_pd = items.toPandas().drop(columns=[f'item_attr_{i}' for i in range(20)])
        cross = users_pd.merge(items_pd, how='cross')
        
        # Convert user and item indices to integers
        cross['user_idx'] = cross['user_idx'].astype(int)
        cross['item_idx'] = cross['item_idx'].astype(int)
        
        # Apply ordinal encoding to category
        cross['category'] = self.ordinal.transform(cross[['category']])
        cross['category'] = cross['category'].astype(int)
        
        # Scale price
        cross['scaled_price'] = self.scaler.transform(cross[['price']])
        
        # Store predictions for each user in a list
        user_predictions = []
        
        # Iterate through each user in the transaction log
        unique_users = log_pd['user_idx'].unique()
        for user in unique_users:
            # Match the user in the transaction log
            user_log = log_pd[log_pd['user_idx']==user]
            # If the user does not appear in the transaction log,
            # use the autoregressive model intercept as the price prediction
            # and a uniform category prediction
            if user_log.empty:
                user_predictions.append([user, self.price_model.intercept_, 0.25, 0.25, 0.25, 0.25])
            else:
                # Otherwise, get the prices of all transactions
                all_user_prices = user_log['scaled_price'].to_numpy()
                
                # Take the 'max_lag' most recent transactions for the price autoregressor
                user_price_X = all_user_prices[-self.max_lag:]
                # Zero-pad this array if its length is less than 'max_lag'
                if len(user_price_X) < self.max_lag:
                    user_price_X = np.pad(
                        user_price_X,
                        (self.max_lag-len(user_price_X),0),
                        'constant',
                        constant_values=0
                    )
                
                # Predict price using the price autoregressor
                user_predicted_price = self.price_model.predict(
                    np.array([user_price_X])
                )
                
                # Take the last transaction for the Markov model
                user_categories = user_log['category'].to_numpy()
                user_last_category = user_categories[-1]
                # Use the values of the Markov column-stochastic matrix from the corresponding column
                user_predicted_category = self.markov[:,user_last_category]
                
                # Add predictions for this user
                user_predictions.append([user, *user_predicted_price, *user_predicted_category])
        
        # Convert user_predictions to DataFrame and add column labels
        user_predictions = pd.DataFrame(user_predictions,
            columns=['user_idx','pred_scaled_price','pred_cat0','pred_cat1','pred_cat2','pred_cat3']
        )
        
        print(user_predictions)
        
        # Merge user_predictions with cross
        cross = cross.merge(user_predictions, on='user_idx')
          
        print(cross)
        
        # Calculate cat_match_prob from selecting the pred_cat value corresponding to category
        cross['cat0'] = 1.0*(cross['category']==0)
        cross['cat1'] = 1.0*(cross['category']==1)
        cross['cat2'] = 1.0*(cross['category']==2)
        cross['cat3'] = 1.0*(cross['category']==3)
        cross['cat_match_prob'] = (
            cross['pred_cat0'] * cross['cat0']
            + cross['pred_cat1'] * cross['cat1']
            + cross['pred_cat2'] * cross['cat2']
            + cross['pred_cat3'] * cross['cat3']
        )
        
        # Calculate price_match_prob from Gaussian PDF with mean 0 and variance 2
        cross['price_diff'] = cross['scaled_price'] - cross['pred_scaled_price']
        cross['price_match_prob'] = np.exp(-np.power(cross['price_diff'],2)/4) / (2 * np.sqrt(np.pi))
        
        # Calculate relevance as category_match_prob * price_match_prob; scale to have maximum of 1
        cross['relevance'] = cross['cat_match_prob'] * cross['price_match_prob']
        cross['relevance'] = cross['relevance'] / max(cross['relevance'])
        # Calculate recommendation score as relevance * price
        cross['score'] = cross['relevance'] * cross['price']
        
        # Group recommendations by user and keep only the best k
        cross = cross.sort_values(by=['user_idx', 'score'], ascending=[True, False])
        cross = cross.groupby('user_idx').head(k)
        
        print(cross)
        
        # Return results as a Spark DataFrame
        return pandas_to_spark(cross[['user_idx', 'item_idx', 'relevance']])