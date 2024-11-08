import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self):
        self.movie_data = None
        self.movie_genres = None
        self.user_profile = None
        self.movie_titles = None
        self.load_data()

    def load_data(self):
        # Wczytanie informacji o filmach
        movies = pd.read_csv('data/movielens/u.item', sep='|', encoding='latin-1', header=None)
        movies = movies.iloc[:, :24]  # Używamy pierwszych 24 kolumn
        movies.columns = ['item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL'] + [f'genre_{i}' for i in
                                                                                                   range(19)]
        # Mapa ID filmu na tytuł
        self.movie_titles = dict(zip(movies['item_id'], movies['title']))
        # Zachowujemy tylko ID, tytuł i gatunki
        self.movie_data = movies[['item_id', 'title'] + [f'genre_{i}' for i in range(19)]]
        # Tworzymy macierz cech gatunków
        self.movie_genres = self.movie_data.set_index('item_id').iloc[:, 1:]

    def build_user_profile(self, user_id):
        # Wczytanie ocen
        ratings = pd.read_csv('data/movielens/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
        user_ratings = ratings[ratings['user_id'] == user_id]
        # Połączenie z gatunkami
        user_movies = self.movie_genres.loc[user_ratings['item_id']]
        user_ratings = user_ratings.set_index('item_id')
        # Obliczenie profilu użytkownika
        self.user_profile = user_movies.mul(user_ratings['rating'], axis=0).sum()

    def recommend(self, user_id, num_recommendations=5):
        # Sprawdzenie, czy użytkownik jest w zbiorze
        ratings = pd.read_csv('data/movielens/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
        if user_id not in ratings['user_id'].unique():
            return []

        self.build_user_profile(user_id)
        # Obliczenie podobieństwa kosinusowego między profilem użytkownika a filmami
        similarity = cosine_similarity([self.user_profile], self.movie_genres.values)[0]
        # Tworzymy DataFrame z wynikami
        similarity_df = pd.DataFrame({'item_id': self.movie_genres.index, 'similarity': similarity})
        # Usuwamy filmy już ocenione przez użytkownika
        rated_items = ratings[ratings['user_id'] == user_id]['item_id']
        recommendations = similarity_df[~similarity_df['item_id'].isin(rated_items)]
        # Sortujemy i wybieramy top N
        recommendations = recommendations.sort_values('similarity', ascending=False).head(num_recommendations)
        # Pobieramy tytuły filmów
        recommended_movie_ids = recommendations['item_id'].tolist()
        recommended_movie_titles = [self.movie_titles[item_id] for item_id in recommended_movie_ids]
        return recommended_movie_titles
