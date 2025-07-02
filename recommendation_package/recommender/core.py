import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class RecommenderSystem:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the recommender system with a dataset.
        """
        self.df = df.copy()
        self.user_item = self._create_user_item_matrix(self.df)

    def _create_user_item_matrix(self, df, fill_value=0):
        """
        Creates a user-item interaction matrix.
        """
        df_clean = df.dropna(subset=['user_id'])
        df_clean['interaction'] = 1
        return df_clean.pivot_table(index='user_id', columns='article_id',
                                    values='interaction', fill_value=fill_value)

    def get_top_articles(self, n=10) -> list:
        top_ids = self.df['article_id'].value_counts().index[:n]
        return self.df[self.df['article_id'].isin(top_ids)]['title'].drop_duplicates().tolist()

    def get_top_article_ids(self, n=10) -> list:
        return list(self.df['article_id'].value_counts().index[:n])

    def get_user_articles(self, user_id: int):
        article_ids = self.user_item.loc[user_id][self.user_item.loc[user_id] == 1].index.tolist()
        article_ids = [int(float(aid)) for aid in article_ids]
        article_names = self.get_article_names(article_ids)
        return article_ids, article_names

    def get_article_names(self, article_ids: list) -> list:
        article_ids = [int(float(aid)) for aid in article_ids]
        article_map = self.df.drop_duplicates('article_id')[['article_id', 'title']]
        id_to_title = dict(zip(article_map['article_id'], article_map['title']))
        return [id_to_title.get(aid) for aid in article_ids if aid in id_to_title]

    def find_similar_users(self, user_id: int, include_similarity=False):
        user_idx = self.user_item.index.get_loc(user_id)
        sim_matrix = cosine_similarity(self.user_item)
        sims = pd.Series(sim_matrix[user_idx], index=self.user_item.index).drop(user_id)
        sims = sims.sort_values(ascending=False)

        if include_similarity:
            return [[int(uid), sim] for uid, sim in sims.items()]
        return sims.index.tolist()

    def get_top_sorted_users(self, user_id: int) -> pd.DataFrame:
        user_idx = self.user_item.index.get_loc(user_id)
        sim_matrix = cosine_similarity(self.user_item)
        sims = sim_matrix[user_idx]

        neighbors_df = pd.DataFrame({
            'neighbor_id': self.user_item.index,
            'similarity': sims,
            'num_interactions': self.user_item.sum(axis=1)
        })
        neighbors_df = neighbors_df[neighbors_df['neighbor_id'] != user_id]
        neighbors_df = neighbors_df.sort_values(by=['similarity', 'num_interactions'], ascending=False)
        return neighbors_df.reset_index(drop=True)

    def user_user_recs(self, user_id: int, m=10) -> list:
        seen_articles, _ = self.get_user_articles(user_id)
        similar_users = self.find_similar_users(user_id)

        recs = []
        for sim_user in similar_users:
            sim_articles, _ = self.get_user_articles(sim_user)
            for aid in sim_articles:
                if aid not in seen_articles and aid not in recs:
                    recs.append(aid)
                    if len(recs) >= m:
                        return recs
        return recs

    def user_user_recs_ranked(self, user_id: int, m=10):
        seen_articles, _ = self.get_user_articles(user_id)
        neighbors_df = self.get_top_sorted_users(user_id)
        recs = []

        for _, row in neighbors_df.iterrows():
            sim_user = row['neighbor_id']
            sim_articles, _ = self.get_user_articles(sim_user)
            for aid in sim_articles:
                if aid not in seen_articles and aid not in recs:
                    recs.append(aid)
                    if len(recs) >= m:
                        break
            if len(recs) >= m:
                break

        if len(recs) > m:
            ranked = self._rank_by_interactions(recs)
            recs = [aid for aid, _ in ranked][:m]

        return recs, self.get_article_names(recs)

    def _rank_by_interactions(self, article_ids: list) -> list:
        counts = [[aid, int(self.user_item[aid].sum())]
                  for aid in article_ids if aid in self.user_item.columns]
        return sorted(counts, key=lambda x: x[1], reverse=True)
