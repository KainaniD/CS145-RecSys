import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as sf
from pyspark.sql.window import Window


def normalize_adj(edge_index, num_nodes):
    
    rows, cols = edge_index
    values = torch.ones(rows.size(0), device=rows.device)
    A = torch.sparse_coo_tensor(torch.stack([rows, cols]), values, (num_nodes, num_nodes))
    deg = torch.sparse.sum(A, dim=1).to_dense()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm_vals = deg_inv_sqrt[rows] * values * deg_inv_sqrt[cols]
    return torch.sparse_coo_tensor(torch.stack([rows, cols]), norm_vals, (num_nodes, num_nodes))


class LightGCNModel(nn.Module):
   
    def __init__(self,
                 num_users: int,
                 num_items: int,
                 embedding_dim: int = 64,
                 num_layers: int = 3,
                 reg_weight: float = 1e-4):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.reg_weight = reg_weight

        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, edge_index):
        A_norm = normalize_adj(edge_index, self.num_nodes)
        x = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        all_emb = x.unsqueeze(1)  # [n,1,d]
        for _ in range(self.num_layers):
            x = torch.sparse.mm(A_norm, x)
            all_emb = torch.cat([all_emb, x.unsqueeze(1)], dim=1)
        final = all_emb.mean(dim=1)
        users, items = final.split([self.num_users, self.num_items], dim=0)
        return users, items

    def bpr_loss(self, users, pos_items, neg_items):
        u = self.user_emb(users)
        i = self.item_emb(pos_items)
        j = self.item_emb(neg_items)
        pos_scores = (u * i).sum(dim=1)
        neg_scores = (u * j).sum(dim=1)
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        reg = self.reg_weight * (
            u.norm(2).pow(2) + i.norm(2).pow(2) + j.norm(2).pow(2)
        ) / users.size(0)
        return loss + reg

    def fit(self,
            edge_index,
            train_edges,
            epochs: int = 10,
            lr: float = 0.01,
            batch_size: int = 1024,
            neg_sampling: int = 1):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        users = train_edges[:, 0]
        items = train_edges[:, 1]
        self.edge_index = edge_index
        for epoch in range(1, epochs + 1):
            perm = torch.randperm(users.size(0))
            u_shuf = users[perm]
            i_shuf = items[perm]
            total_loss = 0.0
            for start in range(0, users.size(0), batch_size):
                batch_u = u_shuf[start:start+batch_size]
                batch_i = i_shuf[start:start+batch_size]
                batch_j = torch.randint(
                    0, self.num_items,
                    (batch_u.size(0),), device=batch_u.device
                )
                optimizer.zero_grad()
                loss = self.bpr_loss(batch_u, batch_i, batch_j)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_u.size(0)

    def predict(self, user_ids, top_k: int = 10, prices=None):
        self.eval()
        with torch.no_grad():
            u_emb, i_emb = self.forward(self.edge_index)
            scores = u_emb[user_ids] @ i_emb.t()
            if prices is not None:
                scores = scores * prices.unsqueeze(0)
            return torch.topk(scores, k=top_k, dim=1).indices


class LightGCNRecommender:
    
    def __init__(self,
                 seed: int = None,
                 emb_dim: int = 64,
                 layers: int = 3,
                 lr: float = 0.01,
                 epochs: int = 10,
                 batch_size: int = 512,
                 neg_sampling: int = 1,
                 reg_weight: float = 1e-4):
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.embedding_dim = emb_dim
        self.num_layers = layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.neg_sampling = neg_sampling
        self.reg_weight = reg_weight
        self.model = None
        self.prices = None

    def fit(self, log: DataFrame, user_features: DataFrame = None, item_features: DataFrame = None):
        # Collect interactions
        log_pd = log.select("user_idx", "item_idx").toPandas()
        users_idx = torch.tensor(log_pd['user_idx'].values, dtype=torch.long)
        items_idx = torch.tensor(log_pd['item_idx'].values, dtype=torch.long)

        # Determine num users/items
        users_pd = user_features.select('user_idx').toPandas()
        items_pd = item_features.select('item_idx', 'price').toPandas()
        num_users = int(users_pd['user_idx'].max()) + 1
        num_items = int(items_pd['item_idx'].max()) + 1

        # Build bipartite graph edges
        src = torch.cat([users_idx, items_idx + num_users])
        dst = torch.cat([items_idx + num_users, users_idx])
        edge_index = torch.stack([src, dst], dim=0)
        train_edges = torch.stack([users_idx, items_idx], dim=1)

        # Prices tensor
        prices_sorted = items_pd.sort_values('item_idx')['price'].values
        self.prices = torch.tensor(prices_sorted, dtype=torch.float)

        # Initialize & train model
        self.model = LightGCNModel(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=self.embedding_dim,
            num_layers=self.num_layers,
            reg_weight=self.reg_weight
        )
        self.model.fit(
            edge_index,
            train_edges,
            epochs=self.epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            neg_sampling=self.neg_sampling
        )

    def predict(self,log: DataFrame,k: int,users: DataFrame,items: DataFrame,user_features: DataFrame = None,item_features: DataFrame = None,filter_seen_items: bool = True) -> DataFrame:
        # Prepare user indices
        users_pd = users.select('user_idx').toPandas()
        user_list = users_pd['user_idx'].astype(int).tolist()
        idx = torch.tensor(user_list, dtype=torch.long)

        # Get top-k item indices by revenue score
        topk = self.model.predict(idx, top_k=k, prices=self.prices)

        # Build recommendation tuples
        recs = []
        for ui, user_idx in enumerate(user_list):
            for item_idx in topk[ui].tolist()[:k]:
                recs.append((int(user_idx), int(item_idx), float(1.0)))

        spark = SparkSession.builder.getOrCreate()
        recs_df = spark.createDataFrame(recs, schema=["user_idx", "item_idx", "relevance"] )

        # Filter seen items
        if filter_seen_items and log is not None:
            seen = log.select("user_idx", "item_idx")
            recs_df = recs_df.join(seen, on=["user_idx", "item_idx"], how="left_anti")

        # Rank & select top-k per user
        window = Window.partitionBy("user_idx").orderBy(sf.desc("relevance"))
        recs_df = recs_df.withColumn("rank", sf.row_number().over(window)).filter(sf.col("rank") <= k).drop("rank")
        return recs_df