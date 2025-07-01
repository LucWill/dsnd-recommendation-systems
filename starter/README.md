# 🧠 IBM Watson Studio Recommendation System

This project implements a **Recommendation System** for the IBM Watson Studio platform. The goal is to enhance the user experience by suggesting relevant articles based on user interactions, article content, and collaborative filtering techniques. The project was developed as part of a data science course module covering unsupervised learning, dimensionality reduction, and recommendation algorithms.

## 🔍 Project Overview

IBM Watson Studio hosts a collaborative ecosystem of articles, datasets, notebooks, and other AI/ML assets. Users interact with these resources, generating valuable behavioral data. This project uses that interaction data to recommend articles to users via:

- **Rank-Based Recommendations** (popularity)
- **Collaborative Filtering** (user-user similarity)
- **Content-Based Filtering** (TF-IDF + KMeans clustering)
- **Matrix Factorization** (SVD with latent features)

## 📁 Project Structure

- `Recommendation_system.ipynb`: Main Jupyter notebook with all tasks and code
- `data/`: Contains interaction data and article content
- `README.md`: Project description and instructions

## 🚀 Technologies Used

- Python (Pandas, NumPy, Scikit-learn)
- Jupyter Notebook
- Natural Language Processing (TF-IDF, cosine similarity)
- Clustering (KMeans)
- Matrix Factorization (SVD)
- IBM Watson Studio context

## 📊 Tasks Breakdown

### 1. Exploratory Data Analysis
- Analyzed user-article interactions
- Calculated statistics (unique users, most viewed articles, etc.)

### 2. Rank-Based Recommendations
- Identified most popular articles based on interaction count
- Provided baseline recommendations for new users

### 3. User-User Collaborative Filtering
- Built a user-item interaction matrix
- Implemented similarity functions to find users with similar reading patterns
- Recommended articles based on neighbors' interactions

### 4. Content-Based Recommendations
- Transformed article titles using TF-IDF
- Reduced dimensionality using Truncated SVD
- Clustered articles using KMeans
- Recommended similar articles based on cluster proximity

### 5. Matrix Factorization
- Performed Singular Value Decomposition (SVD) on the user-item matrix
- Selected optimal number of latent features
- Used cosine similarity on decomposed vectors to recommend similar articles
- Evaluated recommendation performance and discussed improvement strategies

## ✅ Key Features

- Modular and well-documented code
- Functions tested using provided test suites
- DRY principles and reusability through helper functions
- Scalable recommendation architecture

## 🧪 Evaluation and Discussion

The project discusses:
- Strengths and limitations of each recommendation approach
- Suitability of SVD in sparse datasets
- Metrics for real-world recommendation effectiveness
- Potential extensions using deep learning, hybrid methods, and better cold-start handling


## 📚 Acknowledgments

This project was developed as part of a course on **Clustering, Dimensionality Reduction, and Unsupervised Learning**. It builds on interaction data provided in the IBM Watson Studio environment.

---

Feel free to clone, modify, or extend the project!

