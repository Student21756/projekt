import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, precision_score, recall_score

class MatrixFactorizationRecommender:
    def __init__(self):
        self.data = None
        self.user_item_matrix = None
        self.user_ids = None
        self.item_ids = None
        self.movie_titles = None
        self.svd = None
        self.predictions = None
        self.load_data()
        self.train_model()

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

    def train_model(self, n_components=20):
        # Faktoryzacja macierzy
        X = self.user_item_matrix.values
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.svd.fit(X)
        X_pred = self.svd.inverse_transform(self.svd.transform(X))
        self.predictions = pd.DataFrame(X_pred, index=self.user_ids, columns=self.item_ids)

    def recommend(self, user_id, num_recommendations=5):
        # Sprawdzenie, czy użytkownik jest w zbiorze
        if user_id not in self.user_ids:
            return []

        user_ratings = self.user_item_matrix.loc[user_id]
        user_predictions = self.predictions.loc[user_id]

        # Filtracja ocenionych już przedmiotów
        unrated_items = user_ratings[user_ratings == 0].index
        recommendations = user_predictions.loc[unrated_items].sort_values(ascending=False).head(num_recommendations)

        # Konwersja do listy tytułów
        recommended_movie_ids = recommendations.index.tolist()
        recommended_movie_titles = [self.movie_titles[item_id] for item_id in recommended_movie_ids]

        return recommended_movie_titles

    def evaluate(self):
        # Obliczanie RMSE
        X_true = self.user_item_matrix.values[self.user_item_matrix.values.nonzero()].flatten()
        X_pred = self.predictions.values[self.user_item_matrix.values.nonzero()].flatten()
        rmse = np.sqrt(mean_squared_error(X_true, X_pred))
        print(f'RMSE: {rmse}')

        # Obliczanie Precision i Recall
        threshold = 4  # Próg dla pozytywnej oceny
        X_true_binary = (X_true >= threshold).astype(int)
        X_pred_binary = (X_pred >= threshold).astype(int)
        precision = precision_score(X_true_binary, X_pred_binary, zero_division=0)
        recall = recall_score(X_true_binary, X_pred_binary, zero_division=0)
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
