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

    def predict(self, log, k, users, items,
                user_features=None, item_features=None,
                filter_seen_items=True):

        users_pd = users.select('user_idx').toPandas()
        items_pd = items.select('item_idx', 'price')\
                        .toPandas().set_index('item_idx')
        candidate_item_ids = list(items_pd.index)

        user_to_pos = {u: pos for pos, u in enumerate(self.user_idx_list)}
        item_to_pos = {i: pos for pos, i in enumerate(self.item_idx_list)}

        # price vector aligned to self.item_idx_list
        all_price_vector = (
            self.item_price
                .reindex(self.item_idx_list)
                .fillna(0.0)
                .values
        )

        results = []

        for u in users_pd['user_idx']:
            if u not in user_to_pos:
                continue

            u_pos = user_to_pos[u]
            u_feat = self.user_df_scaled.iloc[u_pos].values.reshape(1, -1)
            dists, idxs = self.nn_model.kneighbors(u_feat, n_neighbors=self.k + 1)

            # drop self, keep exactly k neighbours
            neigh_pos = idxs[0].tolist()
            if neigh_pos[0] == u_pos:
                neigh_pos = neigh_pos[1:]
            else:
                neigh_pos = [p for p in neigh_pos if p != u_pos][:self.k]

            neighbour_ids = [self.user_idx_list[p] for p in neigh_pos]

            # count buys among neighbours
            purchase_counts = np.zeros(len(self.item_idx_list), dtype=np.float32)
            for n_uid in neighbour_ids:
                for bought in self.purchase_dict.get(n_uid, ()):
                    if bought in item_to_pos:
                        purchase_counts[item_to_pos[bought]] += 1.0

            p_hat = purchase_counts / float(self.k)
            expected_revenue = p_hat * all_price_vector

            if filter_seen_items:
                for seen in self.purchase_dict.get(u, ()):
                    if seen in item_to_pos:
                        expected_revenue[item_to_pos[seen]] = -np.inf

            # build a temp DataFrame with the raw float scores
            rows = []
            for item_id in candidate_item_ids:
                pos = item_to_pos.get(item_id, None)
                if pos is None:
                    continue
                score = expected_revenue[pos]
                if score == -np.inf:
                    continue
                rows.append({
                    'user_idx': int(u),
                    'item_idx': int(item_id),
                    'expected_revenue': float(score)
                })

            if not rows:
                continue

            df = pd.DataFrame(rows)
            # sort by the float revenue, take top-k 
            df_sorted = df.sort_values('expected_revenue', ascending=False)
            topk = df_sorted.head(k).copy()

            topk['relevance'] = 1.0

            # drop the interim score column
            topk = topk[['user_idx', 'item_idx', 'relevance']]

            results.append(topk)

        if not results:
            empty = pd.DataFrame(columns=['user_idx', 'item_idx', 'relevance'])
            return pandas_to_spark(empty)

        all_recs = pd.concat(results, ignore_index=True)
        return pandas_to_spark(all_recs)