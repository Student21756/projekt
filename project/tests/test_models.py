import unittest
from models.matrix_factorization import MatrixFactorizationRecommender
from models.knn import KNNRecommender

class TestRecommenders(unittest.TestCase):
    def setUp(self):
        self.mf_recommender = MatrixFactorizationRecommender()
        self.knn_recommender = KNNRecommender()

    def test_mf_recommend(self):
        recommendations = self.mf_recommender.recommend(user_id=1)
        self.assertIsInstance(recommendations, list)
        self.assertTrue(len(recommendations) > 0)

    def test_knn_recommend(self):
        recommendations = self.knn_recommender.recommend(user_id=1)
        self.assertIsInstance(recommendations, list)
        self.assertTrue(len(recommendations) > 0)

if __name__ == '__main__':
    unittest.main()
