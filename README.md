# 🧠 IBM Watson Studio Recommendation System

This repository provides a complete, modular implementation of a **Recommendation System** for IBM Watson Studio users, including a reusable Python package and Jupyter-based analysis. It supports ranking, collaborative filtering, content-based recommendations, and matrix factorization techniques.

---

## 📁 Repository Structure

### `recommendation_package/` – Python Package (pip-installable)
```
recommendation_package/
├── recommender/                # Core recommendation modules
│   ├── __init__.py
│   ├── core.py                 # Rank-based + collaborative filtering
│   └── content_based.py        # TF-IDF + clustering recommendations
├── tests/                      # Unit tests
│   ├── test_core.py
│   ├── test_content_based.py
│   └── conftest.py
├── setup.py                    # Packaging configuration
├── README.md                   # ✅ You are here
└── requirements.txt            # Optional dependencies (dev/test)
```

### `starter/` – Analysis & Notebooks
```
starter/
├── data/                                 # Interaction data
│   └── user-item-interactions.csv
├── project_tests.py                      # Udacity test suite
├── Recommendations_with_IBM.ipynb        # Main notebook (without package)
├── Recommendations_with_IBM_using_package.ipynb  # Refactored to use your package
├── Recommendations_with_IBM.pdf          # Final project report
├── top_5.p, top_10.p, top_20.p           # Saved top-N article lists
└── README.md                              # Instructions for the notebook
```

---

## 🚀 Features

✅ Rank-based Recommendations  
✅ User-User Collaborative Filtering  
✅ Content-Based Filtering with TF-IDF and KMeans  
✅ Matrix Factorization using SVD  
✅ Pip-installable Python Package  
✅ Unit Tests with `pytest`  
✅ Ready-to-run Jupyter Notebooks  

---

## 🔧 Installation

Install the package in editable mode from the root of `recommendation_package/`:

```bash
pip install -e .
```

You can then import the modules from any script or notebook:

```python
from recommender.core import get_top_articles, user_user_recs
from recommender.content_based import make_content_recs
```

To run the tests:

```bash
pytest tests/
```

---

## 📚 Project Origin

Developed as part of a **Data Science Nanodegree** module on:

- Clustering and Dimensionality Reduction  
- Unsupervised Learning and Recommendation Systems  
- Practical ML system evaluation  

---

## 📦 Future Extensions

- Deployment as a Streamlit dashboard or Flask app
- Content embeddings with BERT or FastText
- Hybrid models combining user- and content-based signals
- Real-time user feedback integration

---

🧠 Built with Python, Scikit-learn, Pandas, and Jupyter.  
📁 See the `starter/` folder for notebooks and usage examples.
