import pytest
from recommender import RecommenderSystem


def test_create_user_item_matrix(sample_df):
    rec = RecommenderSystem(sample_df)
    user_item = rec.user_item

    # Check dimensions and values
    assert user_item.shape[0] == sample_df['user_id'].nunique()
    assert user_item.shape[1] == sample_df['article_id'].nunique()
    assert (user_item.values.max() <= 1) and (user_item.values.min() >= 0)

def test_get_top_articles(sample_df):
    rec = RecommenderSystem(sample_df)
    top_titles = rec.get_top_articles(n=2)

    assert len(top_titles) == 2
    assert all(isinstance(title, str) for title in top_titles)

def test_get_top_article_ids(sample_df):
    rec = RecommenderSystem(sample_df)
    top_ids = rec.get_top_article_ids(n=2)

    assert len(top_ids) == 2
    assert all(isinstance(aid, (int, float)) for aid in top_ids)

def test_get_user_articles(sample_df):
    rec = RecommenderSystem(sample_df)
    article_ids, article_names = rec.get_user_articles(user_id=1)

    assert isinstance(article_ids, list)
    assert isinstance(article_names, list)
    assert len(article_ids) == len(article_names)

def test_get_article_names(sample_df):
    rec = RecommenderSystem(sample_df)
    names = rec.get_article_names([101, 102])

    assert isinstance(names, list)
    assert all(isinstance(name, str) for name in names)

def test_find_similar_users(sample_df):
    rec = RecommenderSystem(sample_df)
    similar_users = rec.find_similar_users(user_id=1)

    assert isinstance(similar_users, list)
    assert 1 not in similar_users  # target user shouldn't be in list

def test_get_top_sorted_users(sample_df):
    rec = RecommenderSystem(sample_df)
    df = rec.get_top_sorted_users(user_id=1)

    assert 'neighbor_id' in df.columns
    assert 'similarity' in df.columns
    assert 'num_interactions' in df.columns
    assert all(df['neighbor_id'] != 1)

def test_user_user_recs(sample_df):
    rec = RecommenderSystem(sample_df)
    recs = rec.user_user_recs(user_id=1, m=3)

    assert isinstance(recs, list)
    assert len(recs) <= 3
    assert all(isinstance(aid, (int, float)) for aid in recs)

def test_user_user_recs_ranked(sample_df):
    rec = RecommenderSystem(sample_df)
    rec_ids, rec_titles = rec.user_user_recs_ranked(user_id=1, m=3)

    assert isinstance(rec_ids, list)
    assert isinstance(rec_titles, list)
    assert len(rec_ids) <= 3
    assert len(rec_ids) == len(rec_titles)
