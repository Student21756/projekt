import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

class KNNRecommender:
    def __init__(self):
        self.data = None
        self.user_item_matrix = None
        self.user_ids = None
        self.item_ids = None
        self.movie_titles = None
        self.model = None
        self.load_data()
        # Domyślna miara podobieństwa to 'cosine'
        self.train_model(metric='cosine')

    def load_data(self):
        # Wczytanie ocen
        ratings = pd.read_csv('data/movielens/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
        # Wczytanie informacji o filmach
        movies = pd.read_csv('data/movielens/u.item', sep='|', encoding='latin-1', header=None, usecols=[0, 1],
                             names=['item_id', 'title'])
        # Mapa ID filmu na tytuł
        self.movie_titles = dict(zip(movies['item_id'], movies['title']))
        # Tworzenie macierzy użytkownik-przedmiot
        self.user_item_matrix = ratings.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
        self.user_ids = self.user_item_matrix.index.tolist()
        self.item_ids = self.user_item_matrix.columns.tolist()

    def train_model(self, metric='cosine'):
        # Trening modelu k-NN z wybraną miarą podobieństwa
        X = self.user_item_matrix.values
        self.model = NearestNeighbors(metric=metric, algorithm='brute')
        self.model.fit(X)

    def recommend(self, user_id, num_recommendations=5, metric='cosine'):
        # Sprawdzenie, czy model został przetrenowany z daną miarą
        if self.model is None or self.model.effective_metric_ != metric:
            self.train_model(metric=metric)

        # Sprawdzenie, czy użytkownik jest w zbiorze
        if user_id not in self.user_ids:
            return []

        user_index = self.user_ids.index(user_id)
        distances, indices = self.model.kneighbors(self.user_item_matrix.values[user_index].reshape(1, -1), n_neighbors=6)
        similar_users = [self.user_ids[i] for i in indices.flatten() if i != user_index]

        # Agregacja ocen od podobnych użytkowników
        similar_users_ratings = self.user_item_matrix.loc[similar_users]
        mean_ratings = similar_users_ratings.mean(axis=0)
        user_ratings = self.user_item_matrix.loc[user_id]

        # Filtracja ocenionych już przedmiotów
        unrated_items = user_ratings[user_ratings == 0].index
        recommendations = mean_ratings.loc[unrated_items].sort_values(ascending=False).head(num_recommendations)

        # Konwersja do listy tytułów
        recommended_movie_ids = recommendations.index.tolist()
        recommended_movie_titles = [self.movie_titles[item_id] for item_id in recommended_movie_ids]

        return recommended_movie_titles
