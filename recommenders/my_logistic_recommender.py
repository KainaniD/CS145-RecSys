# recommenders/my_logistic_recommender.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as sf
from pyspark.sql.window import Window

class LogisticRecommender:
    def __init__(self, seed=None):
        self.seed = seed
        self.model = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', max_iter=500)
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.scaler = StandardScaler()
        self.trained = False

    def fit(self, log, user_features=None, item_features=None):
        """
        train logistic regression using user and item features + log
        """
        # Convert Spark DataFrames to pandas for easier feature processing and model training
        log_pd = log.toPandas()
        users_pd = user_features.toPandas()
        items_pd = item_features.toPandas()

        # Merge everything
        merged = log_pd.merge(users_pd, on='user_idx').merge(items_pd, on='item_idx')

        # Use relevance as binary label
        y = merged['relevance']

        # Select categorical + numerical features
        X_cat = merged[['segment', 'category']]
        X_num = merged[['price']]

        # Apply one-hot encoding to categorical features and standard scaling to numerical features
        X_cat_encoded = self.encoder.fit_transform(X_cat)
        X_num_scaled = self.scaler.fit_transform(X_num)

        # Combine one-hot encoded features into a single feature matrix.
        X = np.hstack([X_cat_encoded, X_num_scaled])

        self.model.fit(X, y)
        self.trained = True

    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        """
        predict top-k items per user using prob * price
        """
        users_pd = user_features.toPandas()
        items_pd = item_features.toPandas()

        # Create all possible user-item pairs for scoring by performing a cross join
        cross = users_pd.merge(items_pd, how='cross')

        # Keep the original price for scoring and ensure user/item indices are integers
        cross['orig_price'] = cross['price']
        cross['user_idx'] = cross['user_idx'].astype(int)
        cross['item_idx'] = cross['item_idx'].astype(int)

        # Extract features: categorical (user/item type) and numerical (price)
        X_cat = cross[['segment', 'category']]
        X_num = cross[['price']]

        X_cat_encoded = self.encoder.transform(X_cat)
        X_num_scaled = self.scaler.transform(X_num)

        X = np.hstack([X_cat_encoded, X_num_scaled])

        # Predict the probabilities
        probs = self.model.predict_proba(X)[:, 1]

        # Prioritizes items likely to be bought AND generate higher revenue.
        cross['relevance'] = probs * cross['orig_price']

        spark = SparkSession.builder.getOrCreate()

        # 'cross' holds all user-item pairs + features and predicted relevance (prob x price).
        # convert to Spark DataFrame since the simulator expects Spark format for evaluation.
        recs = spark.createDataFrame(cross[['user_idx', 'item_idx', 'relevance']])

        # Filter out items the user has already seen so we don't recommend duplicates.
        if filter_seen_items and log is not None:
            seen = log.select("user_idx", "item_idx")
            recs = recs.join(seen, on=["user_idx", "item_idx"], how="left_anti")

        # For each user, we rank all candidate items by predicted relevance (highest first),
        # then keep only the top-k items per user as final recommendations
        window = Window.partitionBy("user_idx").orderBy(sf.desc("relevance"))
        recs = recs.withColumn("rank", sf.row_number().over(window))
        recs = recs.filter(sf.col("rank") <= k).drop("rank")

        return recs