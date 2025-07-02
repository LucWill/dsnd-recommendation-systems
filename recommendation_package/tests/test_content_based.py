import pytest
from recommender import ContentBasedRecommender


def test_get_similar_articles(sample_df, user_item):
    rec = ContentBasedRecommender(sample_df, user_item)
    similar = rec.get_similar_articles(102)
    assert 103 in similar
    assert 102 not in similar
    assert isinstance(similar, list)

def test_get_ranked_article_unique_counts(sample_df, user_item):
    rec = ContentBasedRecommender(sample_df, user_item)
    counts = rec.get_ranked_article_unique_counts([101, 102, 103])
    assert counts[0][1] >= counts[1][1]
    assert all(isinstance(pair, list) for pair in counts)

def test_get_article_names(sample_df, user_item):
    rec = ContentBasedRecommender(sample_df, user_item)
    names = rec.get_article_names([101, 103])
    assert names == ['A', 'C']

def test_make_content_recs(sample_df, user_item):
    rec = ContentBasedRecommender(sample_df, user_item)
    rec_ids, rec_names = rec.make_content_recs(article_id=102, n=2)
    assert len(rec_ids) <= 2
    assert isinstance(rec_names, list)

def test_get_svd_similar_articles(sample_df, user_item, vt_matrix):
    rec = ContentBasedRecommender(sample_df, user_item, vt_matrix)
    similar = rec.get_svd_similar_articles(article_id=102, top_n=2)
    assert isinstance(similar, list)
    assert 102 not in similar
