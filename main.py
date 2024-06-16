from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

app = Flask(__name__, template_folder='Tamplate')

# Load the trained model for food recommendation
model = load_model('modelrecom.h5')
# Load the datasets
data_restaurant = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTyCBQxA1HyRnQWj-RXf-_5DXKn3L90Td0aO5r8DsTtE7pO3IQbPVvcuhfaCmJvTwiD_Hhvw34Xzbya/pub?gid=2129966204&single=true&output=csv"
data_makanminum = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTyCBQxA1HyRnQWj-RXf-_5DXKn3L90Td0aO5r8DsTtE7pO3IQbPVvcuhfaCmJvTwiD_Hhvw34Xzbya/pub?gid=477290477&single=true&output=csv"

data_resto = pd.read_csv(data_restaurant)
data_makmin = pd.read_csv(data_makanminum)

data_makmin.columns = data_makmin.columns.str.strip()
columns_to_select = ['makanan/minuman', 'kategori', 'deskripsi_rasa']
makanan_data = data_makmin[columns_to_select]

data_resto.columns = data_resto.columns.str.strip()
columns_select = ['nama_restoran', 'rating_toko', 'variasi_makanan','Lokasi']
restoran_data = data_resto[columns_select]

# Combine category and taste for content-based filtering
makanan_data['Features'] = makanan_data['kategori'] + ' ' + makanan_data['deskripsi_rasa']

# Encode the text data using TF-IDF
tfidf = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf.fit_transform(makanan_data['Features'])

# MULAI
# sementara ambil buat model lagi
makanan_data = pd.DataFrame({
    'kategori': data_makmin['kategori'],
    'deskripsi_rasa': data_makmin['deskripsi_rasa'],
    'makanan/minuman': data_makmin['makanan/minuman']
})
# Encode label
label_encoder = LabelEncoder()
label_encoder.fit(makanan_data['kategori'])

# Vectorizer
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(makanan_data['deskripsi_rasa'])

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train_vec.shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])
# END sementara ambil buat model lagi
import json
# END route sementara rekomendasi makanan/minuman
# Fungsi untuk merekomendasikan makanan/minuman bedasarkan inputan
def recommend_makanan(category, taste):
    try:
        # Vektorisasi input menggunakan TfidfVectorizer
        input_vec = vectorizer.transform([taste])

        # Prediksi kategori (dummy predict, ganti dengan metode predict yang sesuai)
        category_pred = np.array([1])  # Dummy prediction
        predicted_category = "Makanan Berat"  # Dummy category

        # Filter data berdasarkan kategori yang diprediksi
        category_data = makanan_data[makanan_data['kategori'] == predicted_category]

        # Vektorisasi deskripsi rasa dalam data latih
        X_train_category = vectorizer.transform(category_data['deskripsi_rasa'])

        # Hitung similarity scores antara input dan deskripsi makanan dalam data latih
        similarity_scores = cosine_similarity(input_vec, X_train_category)

        # Dapatkan indeks top 5 makanan yang paling mirip
        top_indices = similarity_scores.argsort()[0][-5:][::-1]

        # Dapatkan rekomendasi makanan berdasarkan kategori yang diprediksi
        recommendations = []
        for idx in top_indices:
            row = category_data.iloc[idx]
            recommendation = {
                'Makanan': row['makanan/minuman'],
                'similarity_score': float(similarity_scores[0][idx]),
                'kategori': row['kategori'],
                'deskripsi_rasa': row['deskripsi_rasa']
            }
            recommendations.append(recommendation)

        # Kembalikan hasil dalam bentuk JSON
        return json.dumps(recommendations, ensure_ascii=False)
    
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

# Contoh penggunaan
category_input = 'Minuman'
taste_input = 'Segar'
recommended = recommend_makanan(category_input, taste_input)
print("Rekomendasi makanan/minuman:", recommended)

@app.route('/recommend_food', methods=['POST'])
def recommend_food():
    try:
        # data = request.json
        # category = data.get('category')
        # taste = data.get('taste')
        category = request.args.get('category')
        taste = request.args.get('taste')
        if not category or not taste:
            return jsonify({"error": "Category and taste are required"}), 400
        recommendations = recommend_makanan(category, taste)
        
        return jsonify(recommendations)
    except Exception as e:
        print(f"Error in /recommend_food: {e}")
        return jsonify({"error":str(e)}),500
# END route sementara rekomendasi makanan/minuman


@app.route('/')
def home():
    return render_template('index.html')





# Encode the text data using TF-IDF for restaurant recommendation
tfidf_matrix_menu = tfidf.fit_transform(makanan_data['makanan/minuman'])

# Replace NaN values with empty string
restoran_data.loc[:, 'variasi_makanan'] = restoran_data['variasi_makanan'].fillna('')

tfidf_matrix_restaurant = tfidf.transform(restoran_data['variasi_makanan'])

cosine_sim = cosine_similarity(tfidf_matrix_menu, tfidf_matrix_restaurant)

def recommend_restaurants_for_food_containing_keyword(keyword):
    try:
        food_indices = makanan_data[makanan_data['makanan/minuman'].str.contains(keyword, case=False, na=False)].index.tolist()
        recommended_restaurants_with_scores = []

        for food_index in food_indices:
            sim_scores = list(enumerate(cosine_sim[food_index]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            top_similar_restaurants = sim_scores[:5]

            for idx, score in top_similar_restaurants:
                recommended_restaurants_with_scores.append((restoran_data.iloc[idx]['nama_restoran'],restoran_data.iloc[idx]['Lokasi'],
                                                            score, makanan_data.iloc[food_index]['makanan/minuman']))

        recommended_restaurants_with_scores.sort(key=lambda x: x[1], reverse=True)
        unique_recommendations = []
        seen_restaurants = set()

        for rec in recommended_restaurants_with_scores:
            if rec[0] not in seen_restaurants:
                seen_restaurants.add(rec[0])
                unique_recommendations.append(rec)

        return unique_recommendations[:5]
    except Exception as e:
        print(f"Error in recommend_restaurants_for_food_containing_keyword: {e}")
        return []

@app.route('/recommend_restaurant', methods=['POST'])
def recommend_restaurant():
    try:
        # data = request.json
        keyword = request.args.get('keyword')
        if not keyword:
            return jsonify({"error": "Keyword is required"}), 400
        recommendations = recommend_restaurants_for_food_containing_keyword(keyword)
        return jsonify(recommendations)
    except Exception as e:
        print(f"Error in /recommend_restaurant: {e}")
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)