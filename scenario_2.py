import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Load Data
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')
df = pd.merge(ratings, movies, on='movieId')

# 2. Create Matrices
user_item_mat = df.pivot_table(index='userId', columns='title', values='rating').fillna(0)
item_user_mat = df.pivot_table(index='title', columns='userId', values='rating').fillna(0)

# 3. Compute Similarities
# User Similarity
user_sim = cosine_similarity(user_item_mat)
user_sim_df = pd.DataFrame(user_sim, index=user_item_mat.index, columns=user_item_mat.index)

# Item Similarity + Time
start_time = time.time()
item_sim = cosine_similarity(item_user_mat)
item_sim_df = pd.DataFrame(item_sim, index=item_user_mat.index, columns=item_user_mat.index)
item_compute_time = time.time() - start_time

# 4. Similar Items Function
def get_similar_items(movie_title, n=10):
    if movie_title not in item_sim_df.columns:
        return f"Movie '{movie_title}' not found."
    return item_sim_df[movie_title].sort_values(ascending=False).iloc[1:n+1]

# 5. User-Based Recommendation
def recommend_user_based(user_id, matrix, sim_df, n=5):
    user_ratings = matrix.loc[user_id]
    scores = sim_df[user_id].dot(matrix)
    sim_sum = np.abs(sim_df[user_id]).sum()
    preds = scores / sim_sum
    preds = preds.drop(user_ratings[user_ratings > 0].index)
    return preds.sort_values(ascending=False).head(n)

# 6. Item-Based Recommendation
def recommend_item_based(user_id, matrix, sim_df, n=5):
    user_ratings = matrix.loc[:, user_id]
    liked = user_ratings[user_ratings >= 4.0].index
    
    recs = pd.Series(dtype='float64')
    for movie in liked:
        sims = sim_df[movie]
        sims = sims.drop(user_ratings.index, errors='ignore')
        recs = pd.concat([recs, sims])
    
    recs = recs.groupby(recs.index).mean()
    return recs.sort_values(ascending=False).head(n)

# 7. Precision@K
def precision_at_k(user_id, matrix, sim_df, k=5):
    actual = matrix.loc[user_id]
    relevant = set(actual[actual >= 4.0].index)
    
    if not relevant:
        return 0
    
    recs = recommend_user_based(user_id, matrix, sim_df, k)
    hits = len(set(recs.index) & relevant)
    return hits / k

# 8. RMSE Evaluation
subset = ratings.sample(200, random_state=42)
actual, pred_user, pred_item = [], [], []

for _, row in subset.iterrows():
    u, m_id, r = int(row['userId']), int(row['movieId']), row['rating']
    title = movies[movies['movieId'] == m_id]['title'].values[0]
    
    if title in item_sim_df.columns and title in user_item_mat.columns:
        actual.append(r)
        
        # Item-based
        u_ratings = item_user_mat.loc[:, u]
        rated = u_ratings[u_ratings > 0].index
        sims = item_sim_df[title][rated]
        if sims.sum() > 0:
            pred_item.append((sims * u_ratings[rated]).sum() / sims.sum())
        else:
            pred_item.append(df['rating'].mean())
        
        # User-based
        m_ratings = user_item_mat.loc[:, title]
        rated_users = m_ratings[m_ratings > 0].index
        sims_u = user_sim_df[u][rated_users]
        if sims_u.sum() > 0:
            pred_user.append((sims_u * m_ratings[rated_users]).sum() / sims_u.sum())
        else:
            pred_user.append(df['rating'].mean())

rmse_user = np.sqrt(mean_squared_error(actual[:len(pred_user)], pred_user))
rmse_item = np.sqrt(mean_squared_error(actual[:len(pred_item)], pred_item))

# 9. Sample Output
test_user = 1
print("Top User-Based Recommendations:\n", recommend_user_based(test_user, user_item_mat, user_sim_df))
print("\nTop Item-Based Recommendations:\n", recommend_item_based(test_user, item_user_mat, item_sim_df))
print("\nTop Similar to Toy Story:\n", get_similar_items('Toy Story (1995)', 5))

# 10. VISUALIZATIONS

# Comparison Chart
plt.figure(figsize=(8, 5))
methods = ['User-Based', 'Item-Based']
rmse_vals = [rmse_user, rmse_item]
sns.barplot(x=methods, y=rmse_vals, hue=methods, palette='coolwarm', legend=False)
plt.title('RMSE Comparison')
plt.ylabel('Error')
for i, v in enumerate(rmse_vals):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
plt.show()

# User-Item Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(user_item_mat.iloc[:30, :30], cmap='YlGnBu')
plt.title('User-Item Matrix')
plt.show()

# Item Similarity Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(item_sim_df.iloc[:20, :20], cmap='coolwarm')
plt.title('Item Similarity Matrix')
plt.show()

# Top Recommendations Chart
recs = recommend_user_based(test_user, user_item_mat, user_sim_df)
recs_df = pd.DataFrame(recs.items(), columns=['Movie', 'Score'])

plt.figure(figsize=(10, 5))
sns.barplot(x='Score', y='Movie', data=recs_df, hue='Movie', palette='viridis', legend=False)
plt.title(f'Top Recommendations for User {test_user}')
plt.show()

# Toy Story Similar Movies
toy_story = item_sim_df['Toy Story (1995)'].sort_values(ascending=False)[1:11]
plt.figure(figsize=(10, 6))
toy_story.plot(kind='barh', color='skyblue')
plt.title('Top Similar Movies to Toy Story')
plt.gca().invert_yaxis()
plt.show()

# Final Output
print(f"\nUser-Based RMSE: {rmse_user:.4f}")
print(f"Item-Based RMSE: {rmse_item:.4f}")
print(f"Item similarity computed in {item_compute_time:.4f} seconds")