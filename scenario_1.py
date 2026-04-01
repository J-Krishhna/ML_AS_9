import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Load the MovieLens dataset
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

# 2. Preprocessing & Data Inspection
# Merging ratings with movie titles for readability
df = pd.merge(ratings, movies, on='movieId')
print(f"Dataset Shape: {df.shape}")
print(df.head())

# 3. Create User-Item Matrix
# Rows are Users, Columns are Movie Titles
user_item_matrix = df.pivot_table(index='userId', columns='title', values='rating')

# 4. Handle missing values (Fill with 0 for similarity calculation)
user_item_matrix_filled = user_item_matrix.fillna(0)

# Calculate Sparsity
n_total_elements = user_item_matrix.shape[0] * user_item_matrix.shape[1]
n_ratings = ratings.shape[0]
sparsity = (1.0 - (n_ratings / n_total_elements)) * 100
print(f"\nMatrix Sparsity: {sparsity:.2f}%")

# 5. Compute similarity between users (Cosine similarity)
user_similarity = cosine_similarity(user_item_matrix_filled)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix_filled.index, columns=user_item_matrix_filled.index)

# 6. Predict ratings logic
def predict_rating(user_id, movie_title, matrix, similarity_df, k=10):
    """Predicts rating for a specific movie using top-k similar users."""
    if movie_title not in matrix.columns:
        return 0
    
    # Similarity scores for the target user
    sim_scores = similarity_df[user_id]
    
    # Get ratings for the movie from all users
    movie_ratings = matrix[movie_title]
    
    # Find indices where users have actually rated this movie
    idx = movie_ratings[movie_ratings.notnull()].index
    
    # Filter similarity scores for users who rated this movie, excluding the user themselves
    relevant_sims = sim_scores[idx].drop(user_id, errors='ignore')
    
    # Get top k similar users
    top_k_users = relevant_sims.sort_values(ascending=False).head(k)
    
    if top_k_users.sum() == 0:
        return 0 # Or global mean rating
    
    # Weighted average of ratings from similar users
    weighted_sum = (top_k_users * matrix.loc[top_k_users.index, movie_title]).sum()
    prediction = weighted_sum / top_k_users.sum()
    
    return prediction

# 7. Generate Top-N recommendations for a user
def get_recommendations(user_id, matrix, similarity_df, n=5):
    # Find movies the user hasn't seen
    user_ratings = matrix.loc[user_id]
    unseen_movies = user_ratings[user_ratings.isnull()].index
    
    predictions = []
    for movie in unseen_movies:
        pred = predict_rating(user_id, movie, matrix, similarity_df)
        if pred > 0:
            predictions.append((movie, pred))
    
    # Sort by predicted rating
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n]

# Generate recommendations for a sample user (User ID 1)
test_user = 1
recs = get_recommendations(test_user, user_item_matrix, user_similarity_df)
print(f"\nTop Recommendations for User {test_user}:")
for movie, score in recs:
    print(f"- {movie}: {score:.2f}")

# 8. Evaluation (RMSE / MAE)
# Splitting for evaluation: We hide some ratings and try to predict them
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
# (Note: For rigorous evaluation, one should build similarity based ONLY on train_data)

# Sampling test set for performance
test_sample = test_data.sample(100, random_state=42)
actual, predicted = [], []

for _, row in test_sample.iterrows():
    u, m_id, r = int(row['userId']), int(row['movieId']), row['rating']
    m_title = movies[movies['movieId'] == m_id]['title'].values[0]
    
    p = predict_rating(u, m_title, user_item_matrix, user_similarity_df)
    if p > 0:
        actual.append(r)
        predicted.append(p)

print(f"\nEvaluation Metrics:")
print(f"RMSE: {np.sqrt(mean_squared_error(actual, predicted)):.4f}")
print(f"MAE: {mean_absolute_error(actual, predicted):.4f}")

# 9. Visualizations
# Heatmap of User-Item Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(user_item_matrix.iloc[:30, :30], cmap='YlGnBu')
plt.title('User-Item Matrix Heatmap (30x30 Subset)')
# plt.savefig('heatmap_matrix.png')
plt.show()

# Similarity Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(user_similarity_df.iloc[:20, :20], cmap='coolwarm', annot=True)
plt.title('User Similarity Matrix (20x20 Subset)')
# plt.savefig('similarity_matrix.png')
plt.show()

# Top Recommendations Chart
recs_df = pd.DataFrame(recs, columns=['Movie', 'Score'])
plt.figure(figsize=(10, 5))
sns.barplot(x='Score', y='Movie', data=recs_df, hue='Movie', palette='viridis', legend=False)
plt.title(f'Top 5 Recommendations for User {test_user}')
plt.xlabel('Predicted Rating')
# plt.savefig('top_recs_chart.png')
plt.show()