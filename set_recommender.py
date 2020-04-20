#set_recommender.py
from imdb import IMDb
import pandas as pd

def get_movie_genres(movie):
    ia = IMDb()
    movie = ia.search_movie(movie)
    id = movie[0].movieID

    full_movie = ia.get_movie(id)
    genres = full_movie['genres']

    return genres

def build_userprofile(movie_interests, ratings):
    userInput = []
    for idx in range(len(movie_interests)):
        movie_title = movie_interests[idx]
        genres = get_movie_genres(movie_title)
        rating = ratings[idx]

        userInput.append({'title': movie_title, 'rating': rating, 'genres': genres})
    inputMovies = pd.DataFrame(userInput)
    return inputMovies

def get_interests():
    movie_interests = []
    ratings = []

    print("Please enter movies and rating")
    for i in range(5):
        movie = input("Enter movie interest"+str(i)+": ")
        rating = input("Rate 1 to 5: ")
        movie_interests.append(movie)
        ratings.append(rating)

    input_movies = build_userprofile(movie_interests, ratings)
    return input_movies
