from flask import Flask, render_template, request, jsonify
from models.matrix_factorization import MatrixFactorizationRecommender
from models.knn import KNNRecommender
from models.content_based import ContentBasedRecommender

app = Flask(__name__)

# Inicjalizacja modeli
mf_model = MatrixFactorizationRecommender()
knn_model = KNNRecommender()
cb_model = ContentBasedRecommender()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['user_id'])
    algorithm = request.form['algorithm']
    num_recommendations = int(request.form.get('num_recommendations', 5))
    metric = request.form.get('metric', 'cosine')

    if algorithm == 'matrix_factorization':
        recommendations = mf_model.recommend(user_id, num_recommendations=num_recommendations)
    elif algorithm == 'knn':
        recommendations = knn_model.recommend(user_id, num_recommendations=num_recommendations, metric=metric)
    elif algorithm == 'content_based':
        recommendations = cb_model.recommend(user_id, num_recommendations=num_recommendations)
    else:
        return jsonify({'error': 'Nieznany algorytm'}), 400

    return render_template('recommendations.html', recommendations=recommendations)


if __name__ == '__main__':
    app.run(debug=True)
