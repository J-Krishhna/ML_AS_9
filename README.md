# 🎬 Movie Recommendation System (Collaborative Filtering)

This project implements and compares two primary types of Recommendation Engines—User-Based and Item-Based Collaborative Filtering—using the MovieLens dataset to predict user preferences.

---

## 📂 Dataset
**Source:** Kaggle – MovieLens 100k Dataset  
[https://www.kaggle.com/datasets/grouplens/movielens-100k](https://www.kaggle.com/datasets/grouplens/movielens-100k)  
**Key Features:** `userId`, `movieId`, `rating`, `timestamp`, `title`, and `genres`.

---

## 🛠️ Objectives

### Scenario 1: User-Based Collaborative Filtering (UBCF)
- **Data Transformation:** Pivot raw rating logs into a **User-Item Matrix**.
- **Similarity Computation:** Apply **Cosine Similarity** to identify users with similar rating patterns.
- **KNN Logic:** Identify the top-N nearest neighbors for a target user.
- **Rating Prediction:** Use weighted averages of neighbor ratings to predict scores for unseen movies.
- **Evaluation:** Measure performance using **RMSE** (Root Mean Square Error) and **MAE** (Mean Absolute Error).

### Scenario 2: Item-Based Collaborative Filtering (IBCF)
- **Matrix Inversion:** Create an **Item-User Matrix** to focus on movie-to-movie relationships.
- **Item Correlation:** Compute similarity scores between movies based on collective user behavior.
- **Scalability Study:** Measure the time taken to compute the item-similarity matrix versus the user-similarity matrix.
- **Top-N Recommendation:** Generate a list of movies similar to those a user has highly rated (Rating ≥ 4.0).
- **Comparison:** Contrast the accuracy and stability of Item-Based versus User-Based approaches.

---

## 📊 Key Insights & Inferences

### Data Characteristics
- **High Sparsity (98.36%):** The matrix is extremely empty, meaning most users have only rated a tiny fraction of available movies. This leads to "Cold Start" challenges.
- **Long Tail Distribution:** A few popular movies (e.g., *Toy Story*, *Star Wars*) receive the majority of ratings, while niche movies have very few data points.

### Model Performance
- **Accuracy:** In the experimental runs, **Item-Based CF** often yielded a more stable **RMSE (~0.88)** compared to **User-Based CF (~0.92)**, as item-item relationships are less fickle than human tastes.
- **Error Interpretation:** The **RMSE** is higher than **MAE**, indicating that the model occasionally makes large "outlier" errors, likely due to high sparsity in niche movie categories.
- **Stability:** **IBCF** proved more computationally efficient for large-scale application, as movie similarities change less frequently than user preferences.

### Visualization Highlights
- **Similarity Heatmaps:** Visualized the strong correlation between franchise sequels (e.g., *Toy Story* and *Toy Story 2*).
- **Recommendation Comparison:** Bar charts demonstrated that Item-Based filtering provides more intuitive "Because you watched X" style suggestions.
