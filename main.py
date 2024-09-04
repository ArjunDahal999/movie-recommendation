import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import re
from pydantic import BaseModel


app = FastAPI()

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load movie data once at startup
movie_data = pd.read_csv("movies.csv", encoding="utf-8")
movie_data_file = pd.read_csv("movies.csv", encoding="utf-8")
movie_data = movie_data.fillna({
    'budget': 0,
    'genres': 'Unknown',
    'homepage': '',
    'keywords': '',
    'original_language': 'Unknown',
    'original_title': 'Unknown',
    'overview': 'No overview available',
    'popularity': 0.0,
    'production_companies': 'Unknown',
    'production_countries': 'Unknown',
    'release_date': 'Unknown',
    'revenue': 0,
    'runtime': 0.0,
    'spoken_languages': 'Unknown',
    'status': 'Unknown',
    'tagline': '',
    'title': 'Unknown',
    'vote_average': 0.0,
    'vote_count': 0,
    'cast': 'Unknown',
    'crew': 'Unknown',
    'director': 'Unknown'
})
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']



combined_features = movie_data[selected_features].agg(' '.join, axis=1)

# Create TF-IDF vectorizer and similarity matrix
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_vectors)

def predict_movies(movie: str, top_n: int = 15):
    list_of_all_titles = movie_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie, list_of_all_titles, n=1)
    if not find_close_match:
        return []
    close_match = find_close_match[0]
    index_of_the_movie = movie_data[movie_data.title == close_match].index[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    return [movie_data.iloc[movie[0]]['title'] for movie in sorted_similar_movies[1:top_n+1]]


@app.get('/')
def home():
    return {"status": "Working"}


@app.get('/predict/{moviename}')
def predict_movies_endpoint(moviename: str):
    return {"Movies": predict_movies(moviename)}

@app.get('/random-movie')
def random_movie():
    random_movies = movie_data.sample(n=15)[['revenue', 'title', 'director', 'release_date', 'popularity']]
    return {"data": random_movies.to_dict(orient="records")}

@app.get('/top-popular-movies')
def top_popular_movies():
    top_movies = movie_data.nlargest(15, 'popularity')[['revenue', 'title', 'director', 'release_date', 'popularity']]
    return {"data": top_movies.to_dict(orient="records")}

@app.get('/highest-grossing-movie')
def top_popular_movies():
    top_movies = movie_data.nlargest(15, 'revenue')[['revenue', 'title', 'director', 'release_date', 'popularity']]
    return {"data": top_movies.to_dict(orient="records")}


@app.get('/autocomplete/{name}')
def auto_suggestion(name: str):
    if not name:
        return {"name": []}
    pattern = re.compile(re.escape(name), re.IGNORECASE)
    matched_titles = movie_data[movie_data['title'].str.contains(pattern, na=False)]['title'].tolist()
    return {"name": matched_titles[:10]}

@app.get('/{name}')
def get_movie(name: str):
    try:
        result = movie_data.set_index('title').loc[name].drop("crew")
        return {"data": result.to_dict()}
    except KeyError:
        return {"error": "Movie not found"}
