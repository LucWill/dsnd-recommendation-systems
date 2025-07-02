# recommendation_package

A modular Python package for building and evaluating recommendation systems using both **collaborative filtering** and **content-based filtering** approaches. Designed for educational, prototyping, and production-ready use cases.

---

## 🚀 Features

- **Collaborative Filtering** using user-item interaction matrices and cosine similarity
- **Content-Based Recommendations** via title clustering and latent semantic analysis (SVD)
- **Pip-installable modular structure** with clean APIs
- **Unit-tested** core and content-based modules
- Compatible with web app integration (e.g., FastAPI, Streamlit)

---

## 📦 Installation

### Clone and install locally in editable mode:

```bash
git clone https://github.com/LucWill/recommendation_package.git
cd recommendation_package
pip install -e .
```

---

## 🧠 Modules

### `RecommenderSystem`

Located in `recommender/core.py`

Handles collaborative filtering:
- Create user-item matrix
- Get top articles or article IDs
- Find similar users
- Generate user-based recommendations (ranked and unranked)

### `ContentBasedRecommender`

Located in `recommender/content_based.py`

Handles content-based filtering:
- Find articles in the same cluster
- Rank articles by popularity (user interaction count)
- Recommend similar articles via clustering or SVD latent space

---

## 🧪 Testing

Run all tests using `pytest`:

```bash
pytest
```

To check code coverage:

```bash
pytest --cov=recommender tests/
```

---

## 🛠 Example Usage

```python
import pandas as pd
from recommender import RecommenderSystem, ContentBasedRecommender
from sklearn.decomposition import TruncatedSVD

# Load your dataset
df = pd.read_csv('data/articles.csv')

# Collaborative filtering
rec = RecommenderSystem(df)
rec.user_user_recs(user_id=100, m=5)

# Content-based filtering
cb = ContentBasedRecommender(df, rec.user_item)

# Cluster-based recs
cb.make_content_recs(article_id=1320, n=5)

# SVD-based recs
vt = TruncatedSVD(n_components=50).fit_transform(rec.user_item.T)
cb = ContentBasedRecommender(df, rec.user_item, vt=vt)
cb.get_svd_similar_articles(article_id=1320, top_n=5)
```

---

## 📁 Project Structure

```
recommendation_package/
├── recommender/
│   ├── __init__.py
│   ├── core.py                # RecommenderSystem
│   └── content_based.py       # ContentBasedRecommender
├── tests/
│   ├── test_core.py
│   ├── test_content_based.py
│   └── conftest.py
├── setup.py
└── README.md
```

---

## 📜 License

MIT License
