import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as sf
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import DoubleType, ArrayType

from sim4rec.recommenders.ucb import UCB
from sim4rec.utils import pandas_to_spark

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

class KNNRecommender:
    def __init__(self,seed=None, k=10, metric = 'euclidean'):
        self.seed = seed
        self.k = k
        self.metric = metric
        
        # lists of all user/item indexex in training
        self.user_idx_list = []
        self.item_idx_list = []

        # pandas df indexed by user/item index
        self.user_df_scaled = None    
        self.item_df_scaled = None   

        # pandas series indexed by item index
        self.item_price = None  
        
        # maps user -> purchased items
        self.purchase_dict = {}  

        # knn model
        self.nn_model = None         

        self.scalar = StandardScaler()
        
    def fit(self,log,user_features = None, item_features = None):
        
        user_df_full = user_features.toPandas().set_index('user_idx')
        item_df_full = item_features.toPandas().set_index('item_idx')

        # separating price from item features
        if 'price' not in item_df_full.columns:
            raise ValueError("item_features must contain a 'price' column.")

        self.item_price = item_df_full['price'].copy()  
        item_df_no_price = item_df_full.drop(columns=['price'])

        # one-hot encode categorical columns
        user_df_cat = pd.get_dummies(user_df_full)
        item_df_cat = pd.get_dummies(item_df_no_price)

        # fit StandardScaler and transform user and item feature matrices
        self.user_df_scaled = pd.DataFrame(
            self.scalar.fit_transform(user_df_cat.values),
            index=user_df_cat.index,
            columns=user_df_cat.columns
        )

        self.item_df_scaled = pd.DataFrame(
            self.scalar.fit_transform(item_df_cat.values),
            index=item_df_cat.index,
            columns=item_df_cat.columns
        )

        # store sorted lists of user_idx and item_idx 
        self.user_idx_list = list(self.user_df_scaled.index)
        self.item_idx_list = list(self.item_df_scaled.index)

        # map user index -> purchased items
        log_pd = log.select('user_idx', 'item_idx', 'relevance').toPandas()

        # only consider rows with positive relevance as bought/purchased
        log_pd = log_pd[log_pd['relevance'] > 0]
        self.purchase_dict = {u: set() for u in self.user_idx_list}
        for _, row in log_pd.iterrows():
            u = int(row['user_idx'])
            i = int(row['item_idx'])
            if u in self.purchase_dict:
                self.purchase_dict[u].add(i)

        # fit knn
        user_feat_matrix = self.user_df_scaled.values  # (N_users, D_user_features)
        self.nn_model = NearestNeighbors(
            n_neighbors=self.k+1,   # might be k + 1 to include the user itself 
            metric=self.metric,
            algorithm='auto'
        )
        self.nn_model.fit(user_feat_matrix)

    def predict(self,log,k,users,items, user_features = None, item_features = None, filter_seen_items = True):
        users_pd = users.select('user_idx').toPandas()
        items_pd = items.select('item_idx', 'price').toPandas().set_index('item_idx')
        candidate_item_ids = list(items_pd.index)

        user_to_pos = {u: pos for pos, u in enumerate(self.user_idx_list)}
        item_to_pos = {i: pos for pos, i in enumerate(self.item_idx_list)}

        # build a price vector aligned to self.item_idx_list
        all_price_vector = self.item_price.reindex(self.item_idx_list).fillna(0.0).values

        results = []  # collect per-user recommendation dataframes

        # for each requested user, compute top-k
        for u in users_pd['user_idx']:
            if u not in user_to_pos:
                # skip unseen or new users
                continue

            u_pos = user_to_pos[u]
            u_feat = self.user_df_scaled.iloc[u_pos].values.reshape(1, -1)  # (1, D_user)

            # retrieve k+1 nearest neighbors (first is the user)
            distances, indices = self.nn_model.kneighbors(u_feat, n_neighbors=self.k + 1)
            neighbor_positions = indices[0].tolist()
            if neighbor_positions[0] == u_pos:
                neighbor_positions = neighbor_positions[1:]
            else:
                neighbor_positions = [p for p in neighbor_positions if p != u_pos][: self.k]

            # count how many neighbors purchased each item in the global list
            neighbor_user_ids = [self.user_idx_list[p] for p in neighbor_positions]
            purchase_counts = np.zeros(len(self.item_idx_list), dtype=np.float32)
            for n_uid in neighbor_user_ids:
                for purchased_item in self.purchase_dict.get(n_uid, set()):
                    if purchased_item in item_to_pos:
                        purchase_counts[item_to_pos[purchased_item]] += 1.0

            # compute purchase probabilities: p_hat[i] = (# of neighbors who bought i) / k
            p_hat = purchase_counts / float(self.k)

            # compute expected revenue: expected_revenue[i] = p_hat[i] * price[i]
            expected_revenue_all = p_hat * all_price_vector

            # filter out items already seen by this user if requested
            if filter_seen_items:
                for seen_item in self.purchase_dict.get(u, set()):
                    if seen_item in item_to_pos:
                        expected_revenue_all[item_to_pos[seen_item]] = -np.inf

            # build a pandas df for candidate items for this user
            user_rows = []
            for item_id in candidate_item_ids:
                if item_id not in item_to_pos:
                    # if candidate item not present in training set, skip
                    continue
                idx = item_to_pos[item_id]
                rel = float(expected_revenue_all[idx])
                if rel == -np.inf:
                    continue
                user_rows.append({'user_idx': int(u), 'item_idx': int(item_id), 'relevance': rel})

            if not user_rows:
                continue

            user_df_candidates = pd.DataFrame(user_rows)
            # sort by relevance descending and take top k
            user_df_candidates = user_df_candidates.sort_values(by='relevance', ascending=False)
            topk_df = user_df_candidates.head(k)
            results.append(topk_df)

        # if no recommendations were generated, return an empty df
        if not results:
            empty_pd = pd.DataFrame(columns=['user_idx', 'item_idx', 'relevance'])
            return pandas_to_spark(empty_pd)

        # cncatenate all user-specific DataFrames
        all_recs_pd = pd.concat(results, ignore_index=True)

        # convert back to spark df
        recs_spark = pandas_to_spark(all_recs_pd)
        return recs_spark