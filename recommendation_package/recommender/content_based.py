import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self, df: pd.DataFrame, user_item: pd.DataFrame, vt: np.ndarray = None):
        """
        Content-based recommender using clusters and/or SVD latent space.

        Args:
            df (pd.DataFrame): DataFrame containing article metadata including 'title_cluster'.
            user_item (pd.DataFrame): User-item interaction matrix (users x articles).
            vt (np.ndarray, optional): TruncatedSVD V^T matrix for articles.
        """
        self.df = df.copy()
        self.user_item = user_item.copy()
        self.vt = vt

    def get_similar_articles(self, article_id: int) -> list:
        """
        Find other articles in the same cluster as the given article_id.
        """
        if 'title_cluster' not in self.df.columns:
            raise ValueError("DataFrame must contain a 'title_cluster' column.")

        cluster = self.df[self.df['article_id'] == article_id]['title_cluster']
        if cluster.empty:
            return []

        cluster_label = cluster.iloc[0]
        cluster_members = self.df[self.df['title_cluster'] == cluster_label]['article_id'].unique().tolist()
        return [aid for aid in cluster_members if aid != article_id]

    def get_ranked_article_unique_counts(self, article_ids: list) -> list:
        """
        Rank articles by number of unique users who interacted with them.
        """
        article_ids = [int(float(aid)) for aid in article_ids]
        counts = [[aid, int(self.user_item[aid].sum())]
                  for aid in article_ids if aid in self.user_item.columns]
        return sorted(counts, key=lambda x: x[1], reverse=True)

    def get_article_names(self, article_ids: list) -> list:
        """
        Get article titles from article_ids.
        """
        article_ids = [int(float(aid)) for aid in article_ids]
        article_map = self.df.drop_duplicates('article_id')[['article_id', 'title']]
        id_to_title = dict(zip(article_map['article_id'], article_map['title']))
        return [id_to_title.get(aid) for aid in article_ids if aid in id_to_title]

    def make_content_recs(self, article_id: int, n: int = 5) -> tuple:
        """
        Recommend n articles from the same cluster, ranked by popularity.
        """
        similar_articles = self.get_similar_articles(article_id)
        ranked = self.get_ranked_article_unique_counts(similar_articles)
        top_ids = [aid for aid, _ in ranked[:n]]
        top_titles = self.get_article_names(top_ids)
        return top_ids, top_titles

    def get_svd_similar_articles(self, article_id: int, top_n: int = 10, include_similarity=False) -> list:
        """
        Recommend articles most similar to the given article_id based on SVD cosine similarity.
        """
        if self.vt is None:
            raise ValueError("VT matrix not provided.")

        if article_id not in self.user_item.columns:
            raise ValueError(f"Article ID {article_id} not found in user-item matrix.")

        article_idx = list(self.user_item.columns).index(article_id)
        article_latent = self.vt.T  # (num_articles, num_features)

        cos_sim = cosine_similarity([article_latent[article_idx]], article_latent).flatten()
        sorted_indices = np.argsort(cos_sim)[::-1]
        sorted_indices = [i for i in sorted_indices if i != article_idx]

        article_ids = list(self.user_item.columns)
        most_similar_ids = [article_ids[i] for i in sorted_indices[:top_n]]

        if include_similarity:
            return [[article_ids[i], cos_sim[i]] for i in sorted_indices[:top_n]]
        return most_similar_ids
