# Ecommerce-Recommendation-System-import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
data = [
 ('User1', 'Product1', 4),
 ('User1', 'Product2', 5),
 ('User2', 'Product1', 3),
 ('User2', 'Product3', 4),
 ('User3', 'Product2', 5),
 ('User3', 'Product3', 4),
]
product_data = [
 ('Product1', 'Description of Product1'),
 ('Product2', 'Description of Product2'),
 ('Product3', 'Description of Product3'),
]
df = pd.DataFrame(data, columns=['user', 'item', 'rating'])
product_df = pd.DataFrame(product_data, columns=['item', 'description'])
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(df[['user', 'item', 'rating']], reader)
trainset, testset = train_test_split(dataset, test_size=0.2)
sim_options = {
 'name': 'cosine',
 'user_based': True,
}
model = KNNBasic(sim_options=sim_options)
model.fit(trainset)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(product_df['description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
def get_collaborative_recommendations(user_id, model, n=5):
 unrated_items = df.loc[df['user'] == user_id]['item'].unique()
 user_predictions = [(item, model.predict(user_id, item).est) for item in unrated_items]
 user_predictions.sort(key=lambda x: x[1], reverse=True)
 return [item for item, _ in user_predictions[:n]]
def get_content_based_recommendations(item_id, cosine_sim_matrix, n=5):
 item_index = product_df.index[product_df['item'] == item_id].tolist()[0]
 similar_items = list(enumerate(cosine_sim_matrix[item_index]))
 similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)[1:n+1]
 similar_item_indices = [i for i, _ in similar_items]
 return list(product_df['item'].iloc[similar_item_indices])
user_id = 'User1'
collaborative_recommendations = get_collaborative_recommendations(user_id, model, n=3)
print(f"Collaborative Filtering Recommendations for {user_id}: 
{collaborative_recommendations}")
item_id = 'Product1'
content_based_recommendations = get_content_based_recommendations(item_id, 
cosine_sim_matrix=cosine_sim, n=3)
print(f"Content-Based Filtering Recommendations for {item_id}: 
{content_based_recommendations}")
