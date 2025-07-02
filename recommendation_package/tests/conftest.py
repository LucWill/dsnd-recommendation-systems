import pandas as pd
import numpy as np
import pytest
from recommender import RecommenderSystem, ContentBasedRecommender

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'user_id': [1, 2, 1, 3, 2, 4],
        'article_id': [101, 101, 102, 102, 103, 104],
        'title': ['A', 'A', 'B', 'B', 'C', 'D'],
        'title_cluster': [0, 0, 1, 1, 1, 2],
        'email': ['a@x.com', 'b@x.com', 'a@x.com', 'c@x.com', 'b@x.com', 'd@x.com']
    })

@pytest.fixture
def user_item(sample_df):
    recommender = RecommenderSystem(sample_df)
    return recommender.user_item

@pytest.fixture
def vt_matrix(user_item):
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=2)
    return svd.fit_transform(user_item.T)
