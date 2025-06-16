import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from sim4rec.utils import pandas_to_spark

class DTClassifier:
    
    def __init__(self, seed=None, max_depth=None, min_samples_leaf=1, full=True):
        """
        Initialize recommender.
        
        Args:
            seed: Random seed for reproducibility
            max_depth: Maximum decision tree depth (optional)
            min_samples_leaf: Minimum number of samples in each tree leaf (optional)
            full: Whether to use the user and item features (true by default)
        """
        self.seed = seed
        
        # Add your initialization logic here
        self.model = DecisionTreeClassifier(
            criterion='entropy',
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=self.seed
        )
        self.full = full
        
        self.onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.scaler = StandardScaler()
        self.trained = False
    
    def fit(self, log, user_features=None, item_features=None):
        """
        Train the recommender model based on interaction history.
        
        Args:
            log: Interaction log with user_idx, item_idx, and relevance columns
            user_features: User features dataframe (optional)
            item_features: Item features dataframe (optional)
        """
        # Implement your training logic here
        # For example:
        #  1. Extract relevant features from user_features and item_features
        #  2. Learn user preferences from the log
        #  3. Build item similarity matrices or latent factor models
        #  4. Store learned parameters for later prediction
        
        # if self.full and not (user_features and item_features):
        #     return
        
        # Convert to pandas DataFrames
        log_pd = log.toPandas()
        users_pd = user_features.toPandas()
        items_pd = item_features.toPandas()
        
        # Merge everything
        merged = log_pd.merge(users_pd, on='user_idx').merge(items_pd, on='item_idx')
        
        # Use relevance as label
        y = merged['relevance']
        
        # Select categorical and numerical features
        X_cat = merged[['category', 'segment']]
        if self.full:
            # Use the user and item features
            X_num = merged[['price']
                    + [f'user_attr_{i}' for i in range(20)]
                    + [f'item_attr_{i}' for i in range(20)]]
        else:
            X_num = merged[['price']]
        
        # Encode categorical features
        X_cat_encoded = self.onehot.fit_transform(X_cat)
        
        # Scale numerical features
        X_num_scaled = self.scaler.fit_transform(X_num)
        
        # Compile all features
        X = np.hstack([X_cat_encoded, X_num_scaled])
        
        # Train the decision tree model
        self.model.fit(X, y)
        self.trained = True
    
    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        """
        Generate recommendations for users.
        
        Args:
            log: Interaction log with user_idx, item_idx, and relevance columns
            k: Number of items to recommend
            users: User dataframe
            items: Item dataframe
            user_features: User features dataframe (optional)
            item_features: Item features dataframe (optional)
            filter_seen_items: Whether to filter already seen items
            
        Returns:
            DataFrame: Recommendations with user_idx, item_idx, and relevance columns
            -> this is a Spark DataFrame, not a pandas DataFrame
        """
        # Implement your recommendation logic here
        # For example:
        #  1. Extract relevant features for prediction
        #  2. Calculate relevance scores for each user-item pair
        #  3. Rank items by relevance and select top-k
        #  4. Return a dataframe with columns: user_idx, item_idx, relevance
        
        if not self.trained:
            return None
        
        # if self.full and not (user_features and item_features):
        #     return None
        
        # Convert to pandas DataFrames
        # users_pd = user_features.toPandas().copy()
        # items_pd = item_features.toPandas().copy()
        users_pd = users.toPandas().copy()
        items_pd = items.toPandas().copy()
        
        # Cross-join users and items
        cross = users_pd.merge(items_pd, how='cross')
        
        # Convert user and item indices to integers
        cross['user_idx'] = cross['user_idx'].astype(int)
        cross['item_idx'] = cross['item_idx'].astype(int)
        
        # Select categorical and numerical features
        X_cat = cross[['category', 'segment']]
        if self.full:
            # Use the user and item features
            X_num = cross[['price']
                    + [f'user_attr_{i}' for i in range(20)]
                    + [f'item_attr_{i}' for i in range(20)]]
        else:
            X_num = cross[['price']]
        
        # Encode categorical features
        X_cat_encoded = self.onehot.transform(X_cat)
        
        # Scale numerical features
        X_num_scaled = self.scaler.transform(X_num)
        
        # Compile all features
        X = np.hstack([X_cat_encoded, X_num_scaled])
        
        # Make probability/relevance predictions based on the features;
        # calculate "score" as probability * original price
        probs = self.model.predict_proba(X)[:, 1]
        cross['relevance'] = probs
        cross['score'] = probs * cross['price']
        
        # Group recommendations by user and keep only the best k
        cross = cross.sort_values(by=['user_idx', 'score'], ascending=[True, False])
        cross = cross.groupby('user_idx').head(k)
        
        # Return results as a Spark DataFrame
        return pandas_to_spark(cross[['user_idx', 'item_idx', 'relevance']])